"""Startup policy checks for the original-HGR integration stages."""


def validate_baseline_backend(config):
    backend = str(config.get("navigation_backend", "legacy"))
    stages = {"dual_dynamic_shadow": "shadow", "dual_dynamic_route_only": "route_only",
              "dual_dynamic_known_space": "known_space",
              "hgr_dual_topo_memory": "memory", "hgr_dual_topo_intent": "intent",
              "hgr_dual_topo_verified": "verified",
              "hgr_dual_topo_adaptive": "adaptive"}
    if backend not in ("legacy", "dual_dynamic", *stages):
        raise ValueError(f"Unknown navigation_backend: {backend}")
    if backend == "legacy":
        return
    navigation = config.get("dual_dynamic_navigation", {})
    stage = str(navigation.get("stage", "shadow")).strip().lower()
    if backend in stages and stage != stages[backend]:
        raise ValueError(f"{backend} requires stage={stages[backend]}")
    if stage not in (
        "shadow", "route_only", "known_space", "memory", "intent", "verified",
        "adaptive"
    ):
        raise ValueError("Unknown dual_dynamic_navigation.stage")
    if stage in ("memory", "intent", "verified", "adaptive"):
        if backend not in (
            "hgr_dual_topo_memory", "hgr_dual_topo_intent",
            "hgr_dual_topo_verified", "hgr_dual_topo_adaptive"
        ):
            raise ValueError("Hypothesis-aware stages require their dedicated backend")
        preserves_stop = stage not in ("verified", "adaptive")
        if navigation.get("preserve_hgr_stop", preserves_stop) is not preserves_stop:
            raise ValueError("Only verified/adaptive may change the HGR stopping protocol")
        selection_mode = str(navigation.get(
            "semantic_selection_mode", "executable_subset"
        )).strip().lower()
        if selection_mode not in ("executable_subset", "original_hgr"):
            raise ValueError("Unknown semantic_selection_mode")
        verifies = stage in ("verified", "adaptive")
        if navigation.get("enable_terminal_verification", verifies) != verifies:
            raise ValueError("Only verified/adaptive may enable terminal verification")
        persistent_default = stage in ("intent", "verified", "adaptive")
        if navigation.get("enable_incremental_preemption", persistent_default) \
                and not navigation.get("enable_persistent_intent", persistent_default):
            raise ValueError("incremental preemption requires persistent intent")
        if stage == "adaptive":
            if not navigation.get("enable_selective_verification", False):
                raise ValueError("adaptive requires selective terminal verification")
            if navigation.get("enable_revisit_cost_gate", False):
                raise ValueError("adaptive excludes the failed revisit cost gate")
            if not navigation.get("enable_persistent_intent", False):
                raise ValueError("adaptive requires bounded persistent intent")
            if int(navigation.get("max_frontier_intent_motion_steps", 0)) < 1:
                raise ValueError("adaptive requires a positive frontier intent window")
    for flag in (("preserve_hgr_selector", "preserve_hgr_stop")
                 if stage in ("shadow", "route_only", "known_space") else ()):
        if navigation.get(flag, True) is not True:
            raise ValueError(f"{stage} requires {flag}=true")
    if stage in ("shadow", "known_space") and navigation.get("object_approach", False):
        raise ValueError(f"{stage} cannot enable object_approach")
    for section in ("hgr_dual_topo", "hgr_topology_fusion", "place_goal_navigation", "hierarchical_navigation"):
        options = config.get(section, {})
        if options.get("enabled", False) or options.get("route_guidance", False) or options.get("object_approach", False):
            raise ValueError(f"dual_dynamic cannot combine with {section}")
    topology = config.get("active_topology", {})
    place = topology.get("place_topology", {})
    if not (topology.get("enabled", False) and topology.get("mode") == "stable_only"
            and place.get("enabled", False)
            and place.get("stage") == ("shadow" if stage == "known_space" else
                "route_only" if stage in ("memory", "intent", "verified", "adaptive") else stage)):
        raise ValueError("dual_dynamic requires stable_only Place topology with the same stage")
    # These experimental policies have hooks outside the main selector branch.
    for section in ("metric_execution", "metric_action_shadow", "direct_first_execution",
                    "revisit_gate", "target_commitment", "dynamic_goal_topology",
                    "dual_layer_planner", "region_topology", "lightweight_confidence"):
        if topology.get(section, {}).get("enabled", False):
            raise ValueError(f"dual_dynamic cannot combine with active_topology.{section}")
