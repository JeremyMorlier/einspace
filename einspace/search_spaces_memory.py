import logging
import time
from collections import OrderedDict
from copy import deepcopy
from random import choice, choices

import psutil
import torch

from einspace.compiler import Compiler
from einspace.layers import computation_module
from einspace.search_spaces import EinSpace
from einspace.utils import (
    SearchSpaceSamplingError,
    millify,
    predict_num_parameters,
)


logger = logging.getLogger(__name__)


class EinSpaceMemoryEfficient(EinSpace):
    """Iterative sampler variant of EinSpace with lower peak stack/memory overhead.

    This class keeps the architecture format identical to EinSpace, but replaces
    recursive sampling with an explicit stack-based sampler.
    """

    def _choose_option(self, options):
        has_comp_module = any(fn.__name__ == "computation_module" for fn in options)
        if not has_comp_module:
            return choice(options)

        num_other = len(options) - 1
        inv_prob = (1 - self.computation_module_prob) / num_other if num_other > 0 else 0
        probs = [
            self.computation_module_prob if fn.__name__ == "computation_module" else inv_prob
            for fn in options
        ]
        return choices(options, weights=probs, k=1)[0]

    def _make_frame(
        self,
        level,
        input_shape,
        other_shape=None,
        input_mode="im",
        other_mode=None,
        input_branching_factor=1,
        last_im_input_shape=None,
        module_depth=0,
        node_to_remove=None,
    ):
        if module_depth >= self.max_module_depth and level in ["network", "first_fn", "second_fn", "inner_fn"]:
            options = [computation_module]
        else:
            options = list(self.available_options[level])

        if node_to_remove is not None:
            try:
                options.remove(node_to_remove["fn"])
            except ValueError:
                pass

        options = self.filter_options_optimized(
            None,
            options,
            level,
            input_shape,
            other_shape,
            input_mode,
            input_branching_factor,
            module_depth,
        )

        return {
            "level": level,
            "input_shape": input_shape,
            "other_shape": other_shape,
            "input_mode": input_mode,
            "other_mode": other_mode,
            "input_branching_factor": input_branching_factor,
            "last_im_input_shape": last_im_input_shape,
            "module_depth": module_depth,
            "options": options,
            "tries": 0,
            "max_tries": min(max(len(options), 1) * 2, 50),
            "stage": "choose",
            "chosen": None,
            "children": None,
            "inner_nodes": None,
            "pending": None,
        }

    def _make_terminal_node(self, frame, chosen):
        input_shape = frame["input_shape"]
        other_shape = frame["other_shape"]
        input_mode = frame["input_mode"]
        other_mode = frame["other_mode"]
        input_branching_factor = frame["input_branching_factor"]
        last_im_input_shape = frame["last_im_input_shape"]
        module_depth = frame["module_depth"]

        if "im2col" in chosen.__name__:
            last_im_input_shape = chosen(**{"input_shape": input_shape}).fold_output_shape

        return OrderedDict(
            {
                "fn": chosen,
                "input_shape": input_shape,
                "other_shape": other_shape,
                "input_mode": input_mode,
                "other_mode": other_mode,
                "input_branching_factor": input_branching_factor,
                "last_im_input_shape": last_im_input_shape,
                "output_shape": self.recurse_shapes(
                    chosen,
                    input_shape,
                    other_shape,
                    last_im_input_shape,
                    input_branching_factor,
                ),
                "output_mode": self.recurse_modes(chosen, input_mode, other_mode),
                "output_branching_factor": self.recurse_branching(chosen, input_branching_factor),
                "depth": module_depth,
                "node_type": "terminal",
            }
        )

    def _next_child_frame(self, frame):
        chosen_name = frame["chosen"].__name__
        md = frame["module_depth"] + 1

        if chosen_name == "sequential_module":
            children = frame["children"]
            if "first_fn" not in children:
                frame["pending"] = "first_fn"
                return self._make_frame(
                    "first_fn",
                    frame["input_shape"],
                    frame["other_shape"],
                    frame["input_mode"],
                    frame["other_mode"],
                    frame["input_branching_factor"],
                    frame["last_im_input_shape"],
                    md,
                )
            if "second_fn" not in children:
                first_fn = children["first_fn"]
                frame["pending"] = "second_fn"
                return self._make_frame(
                    "second_fn",
                    first_fn["output_shape"],
                    None,
                    first_fn["output_mode"],
                    None,
                    frame["input_branching_factor"],
                    frame["last_im_input_shape"],
                    md,
                )
            return None

        if chosen_name == "routing_module":
            children = frame["children"]
            if "prerouting_fn" not in children:
                frame["pending"] = "prerouting_fn"
                return self._make_frame(
                    "prerouting_fn",
                    frame["input_shape"],
                    frame["other_shape"],
                    frame["input_mode"],
                    frame["other_mode"],
                    frame["input_branching_factor"],
                    frame["last_im_input_shape"],
                    md,
                )
            if "inner_fn" not in children:
                prerouting_fn = children["prerouting_fn"]
                frame["pending"] = "inner_fn"
                return self._make_frame(
                    "inner_fn",
                    prerouting_fn["output_shape"],
                    None,
                    prerouting_fn["output_mode"],
                    None,
                    prerouting_fn["output_branching_factor"],
                    frame["last_im_input_shape"],
                    md,
                )
            if "postrouting_fn" not in children:
                prerouting_fn = children["prerouting_fn"]
                inner_fn = children["inner_fn"]
                frame["pending"] = "postrouting_fn"
                return self._make_frame(
                    "postrouting_fn",
                    inner_fn["output_shape"],
                    None,
                    inner_fn["output_mode"],
                    None,
                    inner_fn["output_branching_factor"],
                    prerouting_fn["last_im_input_shape"],
                    md,
                )
            return None

        if chosen_name == "computation_module":
            children = frame["children"]
            if "computation_fn" not in children:
                frame["pending"] = "computation_fn"
                return self._make_frame(
                    "computation_fn",
                    frame["input_shape"],
                    frame["other_shape"],
                    frame["input_mode"],
                    frame["other_mode"],
                    frame["input_branching_factor"],
                    frame["last_im_input_shape"],
                    md,
                )
            return None

        if chosen_name == "branching_module":
            children = frame["children"]
            inner_nodes = frame["inner_nodes"]

            if "branching_fn" not in children:
                frame["pending"] = "branching_fn"
                return self._make_frame(
                    "branching_fn",
                    frame["input_shape"],
                    frame["other_shape"],
                    frame["input_mode"],
                    frame["other_mode"],
                    frame["input_branching_factor"],
                    frame["last_im_input_shape"],
                    md,
                )

            branching_fn = children["branching_fn"]
            bfactor = self.branching_factor_dict[branching_fn["fn"].__name__]

            if bfactor == 2:
                if len(inner_nodes) < 2:
                    frame["pending"] = "inner_fn"
                    return self._make_frame(
                        "inner_fn",
                        branching_fn["output_shape"],
                        None,
                        branching_fn["output_mode"],
                        None,
                        branching_fn["output_branching_factor"],
                        frame["last_im_input_shape"],
                        md,
                    )
            elif bfactor > 2:
                if len(inner_nodes) == 0:
                    frame["pending"] = "inner_fn"
                    return self._make_frame(
                        "inner_fn",
                        branching_fn["output_shape"],
                        None,
                        branching_fn["output_mode"],
                        None,
                        branching_fn["output_branching_factor"],
                        frame["last_im_input_shape"],
                        md,
                    )
                if len(inner_nodes) != bfactor:
                    template = inner_nodes[0]
                    frame["inner_nodes"] = [deepcopy(template) for _ in range(bfactor)]
                    inner_nodes = frame["inner_nodes"]
            else:
                raise SearchSpaceSamplingError(
                    "A branching factor of 1 is not supported in a branching module."
                )

            if "aggregation_fn" not in children:
                frame["pending"] = "aggregation_fn"
                return self._make_frame(
                    "aggregation_fn",
                    inner_nodes[0]["output_shape"],
                    inner_nodes[1]["output_shape"],
                    inner_nodes[0]["output_mode"],
                    inner_nodes[1]["output_mode"],
                    inner_nodes[0]["output_branching_factor"],
                    frame["last_im_input_shape"],
                    md,
                )
            return None

        raise SearchSpaceSamplingError(f"Unknown nonterminal: {chosen_name}")

    def _attach_child(self, frame, child):
        if frame["pending"] == "inner_fn":
            frame["inner_nodes"].append(child)
        else:
            frame["children"][frame["pending"]] = child
        frame["pending"] = None

    def _finalize_nonterminal_node(self, frame):
        chosen = frame["chosen"]
        chosen_name = chosen.__name__

        base = {
            "fn": chosen,
            "input_shape": frame["input_shape"],
            "other_shape": frame["other_shape"],
            "input_mode": frame["input_mode"],
            "other_mode": frame["other_mode"],
            "input_branching_factor": frame["input_branching_factor"],
            "last_im_input_shape": frame["last_im_input_shape"],
            "depth": frame["module_depth"],
            "node_type": "nonterminal",
        }

        if chosen_name == "sequential_module":
            first_fn = frame["children"]["first_fn"]
            second_fn = frame["children"]["second_fn"]
            return OrderedDict(
                {
                    **base,
                    "children": OrderedDict(
                        {
                            "first_fn": first_fn,
                            "second_fn": second_fn,
                        }
                    ),
                    "output_shape": second_fn["output_shape"],
                    "output_mode": second_fn["output_mode"],
                    "output_branching_factor": second_fn["output_branching_factor"],
                }
            )

        if chosen_name == "routing_module":
            prerouting_fn = frame["children"]["prerouting_fn"]
            inner_fn = frame["children"]["inner_fn"]
            postrouting_fn = frame["children"]["postrouting_fn"]
            return OrderedDict(
                {
                    **base,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": prerouting_fn,
                            "inner_fn": inner_fn,
                            "postrouting_fn": postrouting_fn,
                        }
                    ),
                    "output_shape": postrouting_fn["output_shape"],
                    "output_mode": postrouting_fn["output_mode"],
                    "output_branching_factor": postrouting_fn["output_branching_factor"],
                }
            )

        if chosen_name == "computation_module":
            computation_fn = frame["children"]["computation_fn"]
            return OrderedDict(
                {
                    **base,
                    "children": OrderedDict(
                        {
                            "computation_fn": computation_fn,
                        }
                    ),
                    "output_shape": computation_fn["output_shape"],
                    "output_mode": computation_fn["output_mode"],
                    "output_branching_factor": computation_fn["output_branching_factor"],
                }
            )

        if chosen_name == "branching_module":
            branching_fn = frame["children"]["branching_fn"]
            aggregation_fn = frame["children"]["aggregation_fn"]
            return OrderedDict(
                {
                    **base,
                    "children": OrderedDict(
                        {
                            "branching_fn": branching_fn,
                            "inner_fn": frame["inner_nodes"],
                            "aggregation_fn": aggregation_fn,
                        }
                    ),
                    "output_shape": aggregation_fn["output_shape"],
                    "output_mode": aggregation_fn["output_mode"],
                    "output_branching_factor": aggregation_fn["output_branching_factor"],
                }
            )

        raise SearchSpaceSamplingError(f"Unknown nonterminal: {chosen_name}")

    def sample_iterative(self):
        """Sample one architecture tree using explicit stack-based backtracking."""
        root = self._make_frame("network", self.input_shape, None, self.input_mode, None)
        stack = [root]
        result = None

        while stack:
            if (time.time() - self.start_time) > 60 * 5:
                raise TimeoutError("Sampling took more than 5 minutes. Restarting.")

            frame = stack[-1]

            if frame["stage"] == "choose":
                if not frame["options"]:
                    stack.pop()
                    if not stack:
                        raise SearchSpaceSamplingError(
                            f"No options left to sample from. Level: {frame['level']}."
                        )
                    parent = stack[-1]
                    failed_choice = parent["chosen"]
                    if failed_choice in parent["options"]:
                        parent["options"].remove(failed_choice)
                    parent["stage"] = "choose"
                    parent["chosen"] = None
                    parent["children"] = None
                    parent["inner_nodes"] = None
                    parent["pending"] = None
                    continue

                if frame["tries"] >= frame["max_tries"]:
                    raise SearchSpaceSamplingError(
                        f"Max retries ({frame['max_tries']}) exceeded for {frame['level']}."
                    )

                frame["tries"] += 1
                chosen = self._choose_option(frame["options"])
                frame["chosen"] = chosen

                nonterminal = chosen.__name__ in {
                    "sequential_module",
                    "branching_module",
                    "routing_module",
                    "computation_module",
                }

                if not nonterminal:
                    try:
                        node = self._make_terminal_node(frame, chosen)
                    except Exception:
                        if chosen in frame["options"]:
                            frame["options"].remove(chosen)
                        continue

                    stack.pop()
                    if not stack:
                        result = node
                        break
                    parent = stack[-1]
                    self._attach_child(parent, node)
                    parent["stage"] = "expand"
                    continue

                frame["children"] = OrderedDict()
                frame["inner_nodes"] = []
                frame["pending"] = None
                frame["stage"] = "expand"

            if frame["stage"] == "expand":
                try:
                    child = self._next_child_frame(frame)
                except Exception:
                    failed_choice = frame["chosen"]
                    if failed_choice in frame["options"]:
                        frame["options"].remove(failed_choice)
                    frame["stage"] = "choose"
                    frame["chosen"] = None
                    frame["children"] = None
                    frame["inner_nodes"] = None
                    frame["pending"] = None
                    continue

                if child is not None:
                    stack.append(child)
                    continue

                try:
                    node = self._finalize_nonterminal_node(frame)
                except Exception:
                    failed_choice = frame["chosen"]
                    if failed_choice in frame["options"]:
                        frame["options"].remove(failed_choice)
                    frame["stage"] = "choose"
                    frame["chosen"] = None
                    frame["children"] = None
                    frame["inner_nodes"] = None
                    frame["pending"] = None
                    continue

                stack.pop()
                if not stack:
                    result = node
                    break
                parent = stack[-1]
                self._attach_child(parent, node)
                parent["stage"] = "expand"

        if result is None:
            raise SearchSpaceSamplingError("Failed to sample architecture iteratively.")

        return result

    def sample(self):
        """Sample a random architecture using iterative sampling to reduce recursion overhead."""
        sampling_done = False
        compiler = Compiler()

        memory_usage = psutil.virtual_memory()
        logger.info(
            f"sample(iterative) start: {memory_usage.percent}%, Available: {millify(memory_usage.available, bytes=True)}"
        )

        while not sampling_done:
            self.start_time = time.time()
            try:
                architecture_dict = self.sample_iterative()

                num_predicted_params = predict_num_parameters(architecture_dict)
                available_memory = psutil.virtual_memory().available

                if num_predicted_params < 0.5 * available_memory:
                    modules = compiler.compile(architecture_dict)
                    with torch.no_grad():
                        _ = modules(torch.randn(architecture_dict["input_shape"]))
                    sampling_done = True
                else:
                    logger.info("Architecture too large (parameters estimate), resampling")
            except TimeoutError as e:
                logger.error(f"TimeoutError: {e}")
            except Exception as e:
                logger.error(f"SamplingError: {e}")

        architecture_dict = self.recurse_repeat(architecture_dict, self.num_repeated_cells)
        self.num_nodes = 0
        self.recurse_num_nodes(architecture_dict)
        return architecture_dict
