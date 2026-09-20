import ast
from pathlib import Path
import time
import unittest
from types import SimpleNamespace
import numpy as np
from src.object_approach import resolve_terminal_route
from src.place_topology import PlaceTopology
from src.topology_navigation import PlaceRouteExecution, plan_place_route


class V2RouteStabilityTest(unittest.TestCase):
    @staticmethod
    def main_reset_block():
        source=(Path(__file__).resolve().parents[1]/'run_goatbench_evaluation.py').read_text()
        node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.If)
                  and isinstance(n.test,ast.Attribute) and n.test.attr=='choose_every_step')
        return compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'main-reset','exec')

    def setUp(self):
        self.graph=PlaceTopology(2.,1.,stage='route_only')
        self.a=self.graph.observe_pose([2,3],0).place_id
        self.b=self.graph.observe_pose([8,3],1).place_id
        self.graph.current_place_id=self.a
        self.mask=np.ones((15,15),bool)

    def test_actual_main_passes_active_hypothesis_pin_to_eager_frontier_update(self):
        source=(Path(__file__).resolve().parents[1]/'run_goatbench_evaluation.py').read_text()
        tree=ast.parse(source)
        call=next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'update_frontier_map'
        )
        keyword=next(
            item for item in call.keywords
            if item.arg == 'preserve_hypothesis_node_ids'
        )
        self.assertIsInstance(keyword.value, ast.IfExp)
        self.assertIsInstance(keyword.value.test, ast.Name)
        self.assertEqual(keyword.value.test.id, 'dual_v2_enabled')
        self.assertIsInstance(keyword.value.body, ast.Call)
        self.assertEqual(keyword.value.body.func.attr, 'active_hypothesis_refs')

    def test_valid_approach_survives_cheaper_alternative(self):
        previous={'status':'valid','approach_place_id':self.b}
        result=resolve_terminal_route(self.graph,self.mask,[7,3],[9,3],1.,previous=previous)
        fresh=resolve_terminal_route(self.graph,self.mask,[7,3],[9,3],1.)
        self.assertEqual(fresh['approach_place_id'],self.a)
        self.assertEqual(result['approach_place_id'],self.b)
        self.assertEqual(result['route_event'],'approach_retained')
        self.graph.invalidate_edge(self.a,self.b,2)
        recovered=resolve_terminal_route(self.graph,self.mask,[7,3],[9,3],1.,previous=result)
        self.assertEqual(recovered['approach_place_id'],self.a)
        self.assertEqual(recovered['route_event'],'approach_invalid_reroute')

    def test_actual_main_choose_every_step_keeps_hop_and_clears_on_block(self):
        # Execute the real main-loop reset block, including its retain helper.
        block=self.main_reset_block()
        class Frontier: pass
        frontier=Frontier();frontier.topo_id='f'
        execution=PlaceRouteExecution()
        plan=plan_place_route(self.graph,self.b,'frontier','f',[9,3])
        execution.start(plan)
        planner=SimpleNamespace(max_point=frontier,target_point=plan.target_voxel.copy(),look_at_point=None,
            frontiers=[frontier],island=self.mask,unoccupied=self.mask,occupied=np.zeros_like(self.mask),
            habitat2voxel=lambda p:p)
        env=dict(cfg=SimpleNamespace(choose_every_step=True),object_approach_enabled=True,
            unified_enabled=False,
            dual_v2_enabled=False,dual_v2=None,dual_v2_cfg={},
            place_topology=self.graph,tsdf_planner=planner,place_route_execution=execution,
            place_route_commitment={'target_kind':'frontier','target_id':'f'},pts=np.array([2,3]),
            topology_trace=SimpleNamespace(write=lambda row:None),subtask_id='g',global_step=2,time=time,
            Frontier=Frontier,phase_c_enabled=False,selected_belief_action=None,
            dual_layer_explore_commitment_task_enabled=False)
        exec(block,env)
        self.assertIs(planner.max_point,frontier)
        self.assertIs(execution.active_plan,plan)
        self.assertIsNone(execution.reached_place_id)
        planner.occupied[tuple(np.rint(plan.target_voxel).astype(int))]=True
        exec(block,env)
        self.assertIsNone(planner.target_point)
        self.assertIsNone(planner.max_point)

    def test_actual_main_v2_keeps_cross_place_then_resumes_local_selection(self):
        from src.hgr_dual_topo import DualTopoRuntime
        class Frontier: pass
        frontier=Frontier();frontier.topo_id='f'
        runtime=DualTopoRuntime({})
        runtime.begin_goal('g')
        approach=runtime.resolve(self.graph,self.mask,[2,3],[9,3],1.)
        runtime.install(frontier,approach,self.mask.shape)
        planner=SimpleNamespace(max_point=frontier,target_point=np.array([9,3]),look_at_point=np.array([10,3]))
        env=dict(cfg=SimpleNamespace(choose_every_step=True),object_approach_enabled=True,
            unified_enabled=False,
            dual_v2_enabled=True,dual_v2=runtime,dual_v2_cfg={'persistent_intent':False},
            tsdf_planner=planner,Frontier=Frontier,phase_c_enabled=False,
            selected_belief_action=None,dual_layer_explore_commitment_task_enabled=False)
        block=self.main_reset_block()
        exec(block,env)
        self.assertIs(planner.max_point,frontier)
        self.assertEqual(runtime.execution.remaining_route(),[self.a,self.b])
        point,arrived=runtime.execution.step([2,3],6.,1.,self.mask)
        self.assertFalse(arrived)
        self.assertTrue(runtime.execution.acknowledge(point))
        exec(block,env)
        self.assertIsNone(planner.max_point)
        self.assertIsNone(planner.target_point)
        self.assertIsNone(planner.look_at_point)
        self.assertEqual(self.graph.current_place_id,self.a)

    def test_actual_main_persistent_intent_requires_active_goal(self):
        from src.hgr_dual_topo import DualTopoRuntime
        class Frontier: pass
        frontier=Frontier();frontier.topo_id='f'
        runtime=DualTopoRuntime({'persistent_intent':True})
        runtime.begin_goal('g')
        approach=runtime.resolve(self.graph,self.mask,[2,3],[3,3],1.)
        runtime.install(frontier,approach,self.mask.shape)
        planner=SimpleNamespace(max_point=frontier,target_point=np.array([3,3]),look_at_point=np.array([4,3]))
        env=dict(cfg=SimpleNamespace(choose_every_step=True),object_approach_enabled=True,
            unified_enabled=False,
            dual_v2_enabled=True,dual_v2=runtime,dual_v2_cfg={'persistent_intent':True},
            tsdf_planner=planner,Frontier=Frontier,phase_c_enabled=False,
            selected_belief_action=None,dual_layer_explore_commitment_task_enabled=False)
        block=self.main_reset_block()
        exec(block,env)
        self.assertIs(planner.max_point,frontier)
        runtime.release('source_mapping_invalid')
        exec(block,env)
        self.assertIsNone(planner.target_point)
        self.assertIsNone(planner.max_point)

    def test_arrival_advances_origin_without_changing_observation_place(self):
        previous=resolve_terminal_route(self.graph,self.mask,[2,3],[10,3],1.)
        resumed=resolve_terminal_route(self.graph,self.mask,[8,3],[10,3],1.,origin=self.b,previous=previous)
        self.assertEqual(resumed['route_places'],[self.b])
        self.assertEqual(self.graph.current_place_id,self.a)
