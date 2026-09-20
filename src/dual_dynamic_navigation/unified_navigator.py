"""Single task/intent owner for Memory, persistent execution and verification."""

import copy

import numpy as np

from .baseline_adapter import BaselineDualDynamicNavigator
from .executor import KnownSpaceExecutor, known_mask
from src.route_guidance import grid_path
from .hgr_adapter import HGRAdapter, evidence_digest
from .route_resolver import RouteResolver
from .task_planner import ActionKind, AuthorizedTask, NavigationIntent, NoAction, RequestError, revisit_allowed


class HypothesisAwareNavigator(BaselineDualDynamicNavigator):
    backend_name = "hypothesis_aware_dual_topo"

    def __init__(self, config=None):
        actual = dict(config or {})
        super().__init__({"stage": "route_only"})
        self.config = actual
        self.stage = actual.get("stage", "memory")
        if self.stage not in ("memory", "intent", "verified", "adaptive"):
            raise ValueError("Unknown hypothesis-aware stage")
        self.semantic_selection_mode = str(
            actual.get("semantic_selection_mode", "executable_subset")
        ).strip().lower()
        if self.semantic_selection_mode not in ("executable_subset", "original_hgr"):
            raise ValueError("semantic_selection_mode must be executable_subset or original_hgr")
        self.enable_candidate_annotations = bool(
            actual.get("enable_candidate_annotations", True)
        )
        self.enable_revisit_deduplication = bool(
            actual.get("enable_revisit_deduplication", True)
        )
        self.enable_revisit_cost_gate = bool(
            actual.get("enable_revisit_cost_gate", True)
        )
        self.enable_persistent_intent = bool(actual.get(
            "enable_persistent_intent", self.stage in ("intent", "verified", "adaptive")
        ))
        self.enable_incremental_preemption = bool(actual.get(
            "enable_incremental_preemption", self.stage in ("intent", "verified")
        ))
        self.enable_terminal_verification = bool(actual.get(
            "enable_terminal_verification", self.stage == "verified"
        ))
        self.enable_selective_verification = bool(actual.get(
            "enable_selective_verification", self.stage == "adaptive"
        ))
        self.selective_verification_goal_types = tuple(str(item) for item in actual.get(
            "selective_verification_goal_types", ("image", "description")
        ))
        self.max_frontier_intent_motion_steps = int(actual.get(
            "max_frontier_intent_motion_steps", 0
        ))
        if self.max_frontier_intent_motion_steps < 0:
            raise ValueError("max_frontier_intent_motion_steps must be nonnegative")
        self.historical_confirmation_views = int(actual.get(
            "historical_confirmation_views", 2 if self.stage == "verified" else 1
        ))
        if self.historical_confirmation_views not in (1, 2):
            raise ValueError("historical_confirmation_views must be 1 or 2")
        self.enable_unscored_frontier_fallback = bool(actual.get(
            "enable_unscored_frontier_fallback", True
        ))
        self.adapter, self.resolver = HGRAdapter(), RouteResolver()
        self.executor = KnownSpaceExecutor()
        self.active_intent = None
        self.sequence = 0
        self.checked_views, self.suppressed, self.events = {}, {}, set()
        self.last_selection = None
        self.incremental_seen = set()
        self.incremental_viewpoints = {}
        self.candidate_diagnostics = []
        self.needs_route_revalidation = False

    @property
    def persistent(self):
        return self.enable_persistent_intent

    @property
    def verifies_terminal(self):
        return self.enable_terminal_verification

    def requires_terminal_verification(self, task=None):
        task = task or (None if self.active_intent is None else self.active_intent.task)
        return bool(self.enable_terminal_verification and task is not None
                    and task.requires_verification)

    @property
    def preserves_hgr_policy(self):
        return False

    def begin_goal(self, goal_id):
        if self.needs_route_revalidation and self.goal_id == str(goal_id) and self.active_intent is not None:
            return
        self.release("new_goal")
        super().begin_goal(goal_id)
        self.needs_route_revalidation = False
        self.checked_views, self.suppressed, self.events = {}, {}, set()
        self.last_selection = None
        self.incremental_seen = set()
        self.incremental_viewpoints = {}
        self.candidate_diagnostics = []

    def sync(self, graph, scene, frontiers, event_id, min_detection=1):
        for old, new in getattr(scene, "object_id_aliases", {}).items():
            old, new = str(old), str(new)
            if old == new:
                continue
            self.scene_map.rebind_entity(old, new, f"{event_id}:alias:{old}:{new}")
            self.goal_graph.rebind_entity(old, new, f"{event_id}:alias:{old}:{new}")
            self.checked_views.setdefault(new, []).extend(self.checked_views.pop(old, []))
            if old in self.suppressed:
                self.suppressed[new] = self.suppressed.pop(old)
        return super().sync(graph, scene, frontiers, event_id, min_detection)

    def release(self, reason, hypothesis_graph=None):
        intent = self.active_intent
        if intent is None:
            return None
        if hypothesis_graph is not None:
            self._unpin(hypothesis_graph, intent)
        result = {"intent_id": intent.intent_id, "reason": reason}
        self.active_intent = None
        return result

    @staticmethod
    def _unpin(graph, intent):
        for node_id in intent.pin_refs:
            node = graph.nodes.get(node_id)
            if node is not None:
                owners = set(getattr(node, "navigation_pin_owners", ()))
                owners.discard(intent.intent_id)
                node.navigation_pin_owners = owners
                if not owners and getattr(node, "navigation_pending_removal", False):
                    graph._remove_node(node_id)
        intent.pin_refs = ()

    def install(self, task, hypothesis_graph):
        self.release("replace", hypothesis_graph)
        self.sequence += 1
        intent = NavigationIntent(f"{self.goal_id}:intent:{self.sequence}", self.goal_id, task)
        pins = []
        for node_id in task.hypothesis_refs:
            node = hypothesis_graph.nodes.get(node_id)
            if node is not None:
                owners = set(getattr(node, "navigation_pin_owners", ()))
                owners.add(intent.intent_id)
                node.navigation_pin_owners = owners
                pins.append(node_id)
        intent.pin_refs = tuple(pins)
        self.active_intent = intent
        task_counts = self.stats.setdefault("tasks_by_kind", {})
        task_counts[task.kind.value] = task_counts.get(task.kind.value, 0) + 1
        route_m = self.stats.setdefault("route_distance_m_by_kind", {})
        route_m[task.kind.value] = route_m.get(task.kind.value, 0.0) + float(
            task.route.get("total_distance_m", 0.0)
        )
        return intent

    def configure_task_policy(self, task, metadata):
        if not self.enable_terminal_verification:
            task.requires_verification = False
        elif not self.enable_selective_verification:
            task.requires_verification = task.kind in (
                ActionKind.APPROACH, ActionKind.REVISIT
            )
        else:
            task.requires_verification = bool(
                task.kind == ActionKind.REVISIT
                and self.task_uses_historical_source(task)
                and str(metadata.get("task_type", ""))
                in self.selective_verification_goal_types
            )
        if self.enable_selective_verification and task.requires_verification:
            self.stats["selective_verification_tasks"] = self.stats.get(
                "selective_verification_tasks", 0
            ) + 1
        return task

    def select(self, query, metadata, scene, planner, graph, views, cfg, pts, fresh_objects,
               remaining_steps=None):
        current = planner.habitat2voxel(pts)[:2]
        distance = cfg.planner.final_observe_distance
        fresh = {str(entity) for entity in fresh_objects}
        self.candidate_diagnostics = []
        routes, kinds, digests, eligible = {}, {}, {}, {}
        post_selection_rejections = {}
        frontiers = {}
        for index, frontier in enumerate(planner.frontiers):
            hypothesis = getattr(frontier, "hypothesis_node_id", None)
            if hypothesis is not None:
                node = planner.hypothesis_graph.nodes.get(str(hypothesis))
                if node is None or getattr(getattr(node, "status", None), "value", "") == "invalidated":
                    continue
            route = self.resolver.frontier(graph, planner, frontier, current, distance)
            if route.get("status") == "valid":
                frontiers[index] = route
        explore_costs = [route["total_distance_m"] for route in frontiers.values()]
        def affordable(route, kind):
            if remaining_steps is None:
                return True
            segment = (cfg.planner.max_dist_from_cur_phase_1 if kind == ActionKind.EXPLORE
                       else cfg.planner.max_dist_from_cur_phase_2)
            extra = 1 if (kind == ActionKind.REVISIT or (
                self.verifies_terminal and not self.enable_selective_verification
            )) else 0
            return int(np.ceil(route["total_distance_m"] / segment)) + extra <= remaining_steps
        if self.semantic_selection_mode == "executable_subset":
            frontiers = {i: r for i, r in frontiers.items()
                         if affordable(r, ActionKind.EXPLORE)}
        elif self.enable_revisit_cost_gate:
            for index, route in frontiers.items():
                if not affordable(route, ActionKind.EXPLORE):
                    post_selection_rejections[("frontier", index)] = "remaining_budget"
        explore_costs = [route["total_distance_m"] for route in frontiers.values()]
        for image, snapshot in scene.snapshots.items():
            for obj_id in snapshot.cluster:
                if obj_id not in scene.objects:
                    continue
                entity = self.scene_map.resolve_entity(str(obj_id))
                digest = evidence_digest(scene, image, obj_id)
                diagnostic = {"source_id": str(image), "entity_id": entity,
                              "evidence_digest": digest, "eligible": False}
                self.candidate_diagnostics.append(diagnostic)
                if self.suppressed.get(entity) == digest:
                    diagnostic["reason"] = "suppressed_for_current_evidence"
                    continue
                kind = ActionKind.APPROACH if str(obj_id) in fresh else ActionKind.REVISIT
                route = self.resolver.object_view(graph, planner, scene.objects, obj_id, current,
                    distance, self.checked_views.get(entity, ())
                    if self.enable_revisit_deduplication else ())
                if route.get("status") != "valid":
                    diagnostic["reason"] = route.get("reason", "no_known_route")
                    continue
                if (self.semantic_selection_mode == "executable_subset"
                        and not affordable(route, kind)):
                    diagnostic["reason"] = "remaining_budget"
                    continue
                if (self.semantic_selection_mode == "executable_subset"
                        and self.enable_revisit_cost_gate
                        and kind == ActionKind.REVISIT
                        and not revisit_allowed(route, explore_costs,
                            cfg.planner.max_dist_from_cur_phase_1)):
                    diagnostic["reason"] = "revisit_opportunity_cost"
                    continue
                if (self.semantic_selection_mode == "original_hgr"
                        and self.enable_revisit_cost_gate):
                    if not affordable(route, kind):
                        post_selection_rejections[(str(image), str(obj_id))] = "remaining_budget"
                    elif kind == ActionKind.REVISIT and not revisit_allowed(
                            route, explore_costs, cfg.planner.max_dist_from_cur_phase_1):
                        post_selection_rejections[(str(image), str(obj_id))] = \
                            "revisit_opportunity_cost"
                diagnostic.update(kind=kind.value, eligible=True,
                                  route_cost_m=route["total_distance_m"])
                key = (str(image), str(obj_id))
                routes[key], kinds[key], digests[key] = route, kind, digest
                eligible.setdefault(str(image), []).append(int(obj_id))
        if (self.semantic_selection_mode == "executable_subset"
                and not eligible and not frontiers):
            self.last_selection = NoAction("no_executable_task", tuple(self.checked_views))
            return None
        original_hgr = self.semantic_selection_mode == "original_hgr"
        task_context = ({image: {entity: kinds[(image, entity)].value
            for entity in map(str, ids)} for image, ids in eligible.items()}
            if self.enable_candidate_annotations and not original_hgr else None)
        response = self.adapter.select(
            query, metadata, scene, planner, views, cfg,
            None if original_hgr else eligible,
            None if original_hgr else list(frontiers), task_context,
        )
        if response is None:
            if original_hgr or not frontiers or not self.enable_unscored_frontier_fallback:
                self.last_selection = RequestError("hgr_selection_failed")
                return None
            # Explicit unscored geometry fallback never guesses an object.
            index = min(frontiers, key=lambda i: (frontiers[i]["total_distance_m"], i))
            response = (planner.frontiers[index], 0)
            self.stats["unscored_frontier_fallbacks"] = self.stats.get("unscored_frontier_fallbacks", 0) + 1
        choice, count = response
        cluster = list(getattr(choice, "cluster", ()))
        selected_key = ((str(choice.image), str(cluster[0])) if cluster else
                        ("frontier", next((i for i, frontier in enumerate(planner.frontiers)
                                          if frontier is choice), None)))
        if selected_key in post_selection_rejections:
            gated_eligible = {
                image: [obj_id for obj_id in object_ids
                        if (str(image), str(obj_id)) not in post_selection_rejections]
                for image, object_ids in eligible.items()
            }
            gated_eligible = {image: object_ids for image, object_ids in gated_eligible.items()
                              if object_ids}
            gated_frontiers = [index for index in frontiers
                                if ("frontier", index) not in post_selection_rejections]
            if not gated_eligible and not gated_frontiers:
                self.last_selection = NoAction(
                    post_selection_rejections[selected_key], tuple(self.checked_views)
                )
                return None
            response = self.adapter.select(
                query, metadata, scene, planner, views, cfg,
                gated_eligible, gated_frontiers, None,
                purpose="dual_topo_authorization_reselection",
            )
            self.stats["authorization_reselections"] = self.stats.get(
                "authorization_reselections", 0
            ) + 1
            if response is None:
                self.last_selection = RequestError("authorization_reselection_failed")
                return None
            choice, count = response
        cluster = list(getattr(choice, "cluster", ()))
        if cluster:
            key = (str(choice.image), str(cluster[0]))
            if key not in routes:
                self.stats["selected_sources_not_executable"] = self.stats.get(
                    "selected_sources_not_executable", 0
                ) + 1
                self.last_selection = NoAction(
                    "selected_source_not_executable", (str(choice.image), str(cluster[0]))
                )
                return None
            if key in post_selection_rejections:
                self.last_selection = NoAction(
                    post_selection_rejections[key], (str(choice.image), str(cluster[0]))
                )
                return None
            kind = kinds[key]
            # Historical reuse cannot displace a current object actually selected by HGR.
            task = AuthorizedTask(kind, str(choice.image), self.scene_map.resolve_entity(key[1]),
                                  digests[key], routes[key], choice)
        else:
            index = next((i for i, f in enumerate(planner.frontiers) if f is choice), None)
            if index not in frontiers:
                self.stats["selected_frontiers_not_executable"] = self.stats.get(
                    "selected_frontiers_not_executable", 0
                ) + 1
                self.last_selection = NoAction(
                    "selected_frontier_not_executable",
                    (str(getattr(choice, "topo_id", index)),),
                )
                return None
            if ("frontier", index) in post_selection_rejections:
                self.last_selection = NoAction(
                    post_selection_rejections[("frontier", index)],
                    (str(getattr(choice, "topo_id", index)),),
                )
                return None
            source = str(getattr(choice, "topo_id", index))
            refs = (str(choice.hypothesis_node_id),) if getattr(choice, "hypothesis_node_id", None) else ()
            task = AuthorizedTask(ActionKind.EXPLORE, source, source, "", frontiers[index], choice, refs)
        self.configure_task_policy(task, metadata)
        self.install(task, planner.hypothesis_graph)
        self.last_selection = task
        return choice, count

    def setup(self, planner, pts, cfg):
        intent = self.active_intent
        if intent is None:
            return False
        task = intent.task
        return planner.set_next_navigation_point(task.choice, pts, {}, cfg.planner, None,
            target_point_override=task.route["terminal"], look_at_point=task.route["look_at"],
            known_space_only=True)

    def continue_choice(self, scene, planner, graph=None, pts=None, cfg=None):
        intent = self.active_intent
        if intent is None:
            return None
        if (self.max_frontier_intent_motion_steps
                and intent.task.kind == ActionKind.EXPLORE
                and intent.motion_steps >= self.max_frontier_intent_motion_steps):
            self.stats["bounded_intent_expirations"] = self.stats.get(
                "bounded_intent_expirations", 0
            ) + 1
            self.release("bounded_frontier_intent_expired", planner.hypothesis_graph)
            return None
        if self.needs_route_revalidation and graph is not None:
            from src.object_approach import resolve_terminal_route
            route = resolve_terminal_route(graph, known_mask(planner), planner.habitat2voxel(pts)[:2],
                intent.task.route["terminal"], cfg.planner.final_observe_distance)
            route = self.resolver.certify(route, graph, planner, planner.habitat2voxel(pts)[:2])
            if route.get("status") != "valid":
                self.release("restored_route_blocked", planner.hypothesis_graph)
                return None
            route.update(terminal=intent.task.route["terminal"], look_at=intent.task.route["look_at"])
            intent.task.route = route
            planner.max_point = planner.target_point = planner.look_at_point = None
            self.needs_route_revalidation = False
            if not self.setup(planner, pts, cfg):
                self.release("restored_terminal_invalid", planner.hypothesis_graph)
                return None
        if intent.task.kind == ActionKind.EXPLORE:
            choice = next((f for f in planner.frontiers
                if str(getattr(f, "topo_id", "")) == intent.task.source_id), None)
            node = planner.hypothesis_graph.nodes.get(
                next(iter(intent.task.hypothesis_refs), ""))
            if intent.task.hypothesis_refs and (node is None or
                    getattr(getattr(node, "status", None), "value", "") == "invalidated"):
                self.goal_graph.revoked_hypotheses.update(intent.task.hypothesis_refs)
                self.release("hypothesis_invalid", planner.hypothesis_graph)
                return None
            if choice is not None:
                intent.task.choice = choice
            elif graph is not None and pts is not None:
                previous = intent.task.choice
                direction = np.asarray(previous.orientation[:2], float)
                candidates = []
                for frontier in planner.frontiers:
                    normal = np.asarray(frontier.orientation[:2], float)
                    cosine = float(direction @ normal) / max(np.linalg.norm(direction)*np.linalg.norm(normal), 1e-12)
                    if cosine < np.cos(np.deg2rad(30)):
                        continue
                    route = self.resolver.frontier(graph, planner, frontier,
                        planner.habitat2voxel(pts)[:2], cfg.planner.final_observe_distance)
                    if route.get("status") != "valid":
                        continue
                    connector = grid_path(known_mask(planner), intent.task.route["terminal"], route["terminal"])
                    if connector is None:
                        continue
                    cost = float(np.linalg.norm(np.diff(connector, axis=0), axis=1).sum()*graph.voxel_size)
                    if cost <= graph.place_spacing_m:
                        candidates.append((cost, str(getattr(frontier, "topo_id", "")), frontier, route))
                if candidates:
                    _, source, choice, route = min(candidates, key=lambda item: item[:2])
                    intent.task.choice, intent.task.source_id, intent.task.entity_id = choice, source, source
                    intent.task.route = route
                    planner.target_point = np.asarray(route["terminal"])
                    planner.look_at_point = np.asarray(route["look_at"])
            # Re-extraction alone does not cancel a still-certified physical terminal.
            return intent.task.choice
        aliases = getattr(scene, "object_id_aliases", {})
        object_id = int(intent.task.choice.cluster[0])
        if object_id not in scene.objects:
            new, seen = object_id, set()
            while new in aliases and new not in seen:
                seen.add(new)
                new = aliases[new]
            if new is None or new not in scene.objects:
                self.release("entity_source_invalid", planner.hypothesis_graph)
                return None
            intent.task.choice = copy.copy(intent.task.choice)
            intent.task.choice.cluster = [new]
            self.scene_map.rebind_entity(str(object_id), str(new), f"alias:{object_id}:{new}")
            intent.task.entity_id = str(new)
        return intent.task.choice

    def assess_incremental(self, query, metadata, scene, planner, graph, views, cfg, pts,
                           fresh_objects, current_rgb, verifier, remaining_steps=None):
        intent = self.active_intent
        if (not self.enable_incremental_preemption or not self.persistent
                or intent is None or intent.task.kind == ActionKind.APPROACH):
            return False
        eligible = {}
        for image, snapshot in scene.snapshots.items():
            for entity in snapshot.cluster:
                if entity not in fresh_objects or entity not in scene.objects:
                    continue
                digest = evidence_digest(scene, image, entity)
                key = (self.scene_map.resolve_entity(str(entity)), digest)
                if key in self.incremental_seen:
                    continue
                position = np.asarray(snapshot.obs_point[:2], float)
                previous = self.incremental_viewpoints.setdefault(key[0], [])
                if any(np.linalg.norm(position-point)*graph.voxel_size <= .5 for point in previous):
                    continue
                self.incremental_seen.add(key)
                previous.append(position.tolist())
                eligible.setdefault(str(image), []).append(int(entity))
        if not eligible:
            return False
        self.stats["incremental_assessments"] = self.stats.get("incremental_assessments", 0) + 1
        response = self.adapter.select(query, metadata, scene, planner, views, cfg, eligible, [],
            {image: {str(entity): "APPROACH" for entity in entities} for image, entities in eligible.items()},
            purpose="dual_topo_incremental_selection")
        if response is None:
            return False
        choice, _ = response
        cluster = getattr(choice, "cluster", ())
        if len(cluster) != 1 or cluster[0] not in eligible.get(str(choice.image), ()):
            return False
        verdict, _ = verifier(metadata, current_rgb() if callable(current_rgb) else current_rgb,
                              choice, cfg, purpose="dual_topo_incremental_assessment")
        if verdict != "confirmed":
            return False
        route = self.resolver.object_view(graph, planner, scene.objects, cluster[0],
            planner.habitat2voxel(pts)[:2], cfg.planner.final_observe_distance)
        if route.get("status") != "valid":
            return False
        task = AuthorizedTask(ActionKind.APPROACH, str(choice.image),
            self.scene_map.resolve_entity(str(cluster[0])), evidence_digest(scene, choice.image, cluster[0]),
            route, choice)
        self.configure_task_policy(task, metadata)
        if remaining_steps is not None and int(np.ceil(route["total_distance_m"] /
                cfg.planner.max_dist_from_cur_phase_2)) + int(
                    task.requires_verification
                ) > remaining_steps:
            return False
        self.install(task, planner.hypothesis_graph)
        self.stats["successful_incremental_preemptions"] = self.stats.get(
            "successful_incremental_preemptions", 0
        ) + 1
        planner.max_point = planner.target_point = planner.look_at_point = None
        return self.setup(planner, pts, cfg)

    def feedback(self, verdict, intent_id, event_id, planner):
        intent = self.active_intent
        if (event_id in self.events or intent is None or intent.intent_id != intent_id
                or intent.state == "confirmed"):
            return "ignored"
        self.events.add(event_id)
        task = intent.task
        if verdict not in ("confirmed", "rejected", "uncertain", "error"):
            raise ValueError("Invalid verification verdict")
        self.checked_views.setdefault(task.entity_id, []).append(task.route["terminal"])
        intent.verification_views += 1
        if verdict == "confirmed":
            if (self.task_uses_historical_source(task)
                    and intent.verification_views < self.historical_confirmation_views):
                self.stats["historical_confirmation_deferrals"] = self.stats.get(
                    "historical_confirmation_deferrals", 0
                ) + 1
                return "new_view"
            from src.hierarchical_navigator import VerificationVerdict
            self.goal_graph.record_verification(event_id, task.entity_id,
                (task.evidence_digest,), VerificationVerdict.CONFIRMED)
            if task.kind == ActionKind.REVISIT:
                task.kind = ActionKind.APPROACH
            intent.state = "confirmed"
            return "stop"
        if verdict == "rejected":
            self.suppressed[task.entity_id] = task.evidence_digest
            from src.hierarchical_navigator import VerificationVerdict
            self.goal_graph.record_verification(event_id, task.entity_id,
                (task.evidence_digest,), VerificationVerdict.REJECTED)
        if verdict == "uncertain" and intent.verification_views < 2:
            return "new_view"
        if verdict == "uncertain":
            from src.hierarchical_navigator import VerificationVerdict
            self.goal_graph.record_verification(event_id, task.entity_id,
                (task.evidence_digest,), VerificationVerdict.UNCERTAIN)
        self.release(f"verification_{verdict}", planner.hypothesis_graph)
        return "reselect"

    def task_source_first_seen(self, task):
        entity = self.scene_map.entities.get(
            self.scene_map.resolve_entity(task.entity_id)
        )
        if entity is None:
            return None
        prefix = f"snapshot:{task.source_id}:object:"
        rows = [
            entity.evidence_first_seen[evidence_id]
            for evidence_id in entity.evidence_ids
            if evidence_id.startswith(prefix)
            and evidence_id in entity.evidence_first_seen
        ]
        return None if not rows else min(rows, key=lambda row: int(row[1]))

    def task_uses_historical_source(self, task):
        first_seen = self.task_source_first_seen(task)
        return bool(first_seen is not None and str(first_seen[0]) != str(self.goal_id))

    def promote_revisit(self, intent_id):
        """Promote an arrived historical revisit to the baseline stop-bearing task.

        HGR already selected the source for the current goal.  REVISIT limits that
        source to known-space motion while it is still historical; arrival supplies
        the fresh observation required to restore the original APPROACH contract.
        The same intent and source identity must survive the transition so the
        runner can apply the ordinary snapshot-arrival stopping rule.
        """
        intent = self.active_intent
        if (intent is None or intent.intent_id != intent_id
                or intent.task.kind != ActionKind.REVISIT):
            return None
        task = intent.task
        self.checked_views.setdefault(task.entity_id, []).append(
            task.route["terminal"]
        )
        self.stats["completed_revisits"] = self.stats.get(
            "completed_revisits", 0
        ) + 1
        self.stats["promoted_revisits"] = self.stats.get(
            "promoted_revisits", 0
        ) + 1
        task.kind = ActionKind.APPROACH
        intent.state = "arrived"
        result = {
            "intent_id": intent.intent_id,
            "entity_id": task.entity_id,
            "source_id": task.source_id,
            "evidence_digest": task.evidence_digest,
            "transition": "REVISIT->APPROACH",
            "stop_authorized": True,
        }
        return result

    def complete_revisit(self, intent_id, hypothesis_graph=None):
        """Compatibility alias for checkpoints and callers predating promotion."""
        return self.promote_revisit(intent_id)

    def record_motion_progress(self, planner, graph, pts, moved=True, arrived=False):
        """Detect three comparable motion steps without one-voxel path progress."""
        intent = self.active_intent
        if intent is not None and moved:
            intent.motion_steps += 1
        if intent is None or arrived or not moved:
            if intent is not None and arrived:
                intent.stalled_steps = 0
            return False
        current = np.asarray(planner.habitat2voxel(pts)[:2], float)
        terminal = np.asarray(intent.task.route["terminal"], float)
        path = grid_path(known_mask(planner), current, terminal)
        if path is None:
            return False
        distance_m = float(
            np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * graph.voxel_size
        )
        minimum_progress = float(graph.voxel_size)
        if (intent.last_distance_m is not None
                and intent.last_distance_m - distance_m < minimum_progress - 1e-6):
            intent.stalled_steps += 1
        else:
            intent.stalled_steps = 0
        intent.last_distance_m = distance_m
        if intent.stalled_steps < 3:
            return False
        intent.stalled_steps = 0
        self.stats["stalls_detected"] = self.stats.get("stalls_detected", 0) + 1
        return True

    def recover(self, planner, graph, pts, observe_distance):
        intent = self.active_intent
        if intent is None:
            return False
        current = planner.habitat2voxel(pts)[:2]
        mask = known_mask(planner)
        if intent.recovery_attempts == 0:
            intent.recovery_attempts = 1
            guide = self.executor.guidance
            if guide is not None and guide.repair(mask, current):
                intent.task.route = {**intent.task.route,
                    "certified_path": guide.path.tolist(), "transitions": guide.transitions,
                    "route_places": guide.route, "total_distance_m": float(
                        np.linalg.norm(np.diff(guide.path, axis=0), axis=1).sum()*graph.voxel_size)}
                return True
        if intent.recovery_attempts == 1:
            intent.recovery_attempts = 2
            from src.object_approach import resolve_terminal_route
            route = resolve_terminal_route(graph, mask, current, intent.task.route["terminal"], observe_distance)
            route = self.resolver.certify(route, graph, planner, current)
            if route.get("status") == "valid":
                route.update(terminal=intent.task.route["terminal"], look_at=intent.task.route["look_at"])
                intent.task.route = route
                return True
        return False

    def validate_restore(self, config):
        if self.config != dict(config):
            raise ValueError("Hypothesis-aware checkpoint configuration mismatch")
        self.executor = KnownSpaceExecutor()
        self.needs_route_revalidation = True
        return self

    def checkpoint(self):
        return {"schema_version": 1, "backend": self.backend_name,
                "goal_id": self.goal_id, "config": dict(self.config), "navigation": copy.deepcopy(self)}

    def to_trace_dict(self):
        result = super().to_trace_dict()
        task_trace = None
        if self.active_intent is not None:
            task_trace = self.active_intent.task.trace()
            task = self.active_intent.task
            first_seen = self.task_source_first_seen(task)
            task_trace["source_first_seen_goal"] = (
                None if first_seen is None else str(first_seen[0])
            )
            task_trace["historical_source"] = self.task_uses_historical_source(task)
        result.update(backend=self.backend_name, stage=self.stage,
                      preserves_hgr_selector=False,
                      preserves_hgr_stop=self.stage not in ("verified", "adaptive"),
                      active_intent=None if self.active_intent is None else {
                          "intent_id": self.active_intent.intent_id,
                          "state": self.active_intent.state, "task": task_trace},
                      checked_views=self.checked_views)
        result["selection_outcome"] = (self.last_selection.trace() if isinstance(self.last_selection, AuthorizedTask)
            else None if self.last_selection is None else vars(self.last_selection))
        result["candidates"] = self.candidate_diagnostics
        return result
