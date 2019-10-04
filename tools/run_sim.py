import wx
import gflags
import sys
from rmp_nav.simulation.sim_gui_wx import SimGui
from rmp_nav.simulation.gibson_map import MakeGibsonMap
from rmp_nav.common.utils import get_project_root


gflags.DEFINE_string('scene_id', 'space2', '')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)


# Increase path_map_dilation to make paths further away from obstacles.
# It should be at least 1 to avoid intersecting with obstacles.
# Too large value may cause planning failure.

DEFAULT_PATH_MAP_DILATION = 5
DEFAULT_REACHABLE_AREA_DILATION = 2

gibson_map = MakeGibsonMap(get_project_root() + '/rmp_nav/gibson/assets/dataset/',
                           FLAGS.scene_id,
                           path_map_dilation=DEFAULT_PATH_MAP_DILATION,
                           reachable_area_dilation=DEFAULT_REACHABLE_AREA_DILATION,
                           path_map_weighted_dilation=True)

app = wx.App()


gui = SimGui()

gui.SetPosition((0, 0))

gui.load_map(gibson_map)


gui.set_start_pos(-3.20, -0.59)
gui.set_goal_pos(0.98, 25.47)
gui.set_heading(0.0)


gui.control_panel.goal_destination_mode.SetValue(False)
gui.control_panel.no_goal.SetValue(False)
gui.control_panel.no_planning.SetValue(False)
gui.control_panel.show_traj.SetValue(True)
gui.control_panel.show_control_point_accels.SetValue(True)
gui.control_panel.show_control_point_metrics.SetValue(True)
gui.control_panel.show_obstacles.SetValue(True)

gui.control_panel.agent_checkedbuttons[0].SetValue(True)

gui.Show()
app.MainLoop()
