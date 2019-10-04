from __future__ import print_function
from future.utils import iteritems
import wx
import wx.lib.scrolledpanel

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

import numpy as np
from .map_visualizer import FindMapVisualizer
from .agent_visualizers import FindAgentVisualizer
from . import agent_factory, param_utils

import time
import cv2
import threading
import cairo
from rmp_nav.common.utils import str_to_dict, cairo_argb_to_opencv_bgr


class Plot(wx.Panel):
    def __init__(self, parent, id=-1, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.figure = plt.Figure(figsize=(8, 6), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)


class Controls(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.agent_checkedbuttons = None
        self.agent_preference_buttons = None
        self.agent_preference_texts = None

        self.start_sim_button = None

        self.set_start_pos_button = None
        self.start_pos_text = None
        self.set_goal_pos_button = None
        self.goal_pos_text = None

        self.show_axes_ticks = None
        self.show_title = None
        self.show_legends = None
        self.show_waypoints = None
        self.show_obstacles = None
        self.show_traj = None
        self.show_control_point_metrics = None
        self.show_control_point_accels = None
        self.show_accel = None
        self.stop_on_collision = None

        self.save_screenshot = None

        vbox = wx.BoxSizer(wx.VERTICAL)

        vbox.Add(self._make_simulation_box(),
                 flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10, proportion=0)

        vbox.Add(self._make_agents_box(),
                 flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border=10, proportion=1)

        vbox.Add(self._make_options_box(),
                 flag=wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10, proportion=0)

        self.SetSizer(vbox)

    def _make_agents_box(self):
        sb = wx.StaticBox(self, label='Agents')
        scrolled_panel = wx.lib.scrolledpanel.ScrolledPanel(sb, -1, size=(-1, -1))
        items_sizer = wx.BoxSizer(wx.VERTICAL)

        self.agent_checkedbuttons = []
        self.agent_preference_buttons = []
        self.agent_preference_texts = []

        for idx, agent_constructor in enumerate(agent_factory.public_agents):
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            checkbox = wx.CheckBox(scrolled_panel, label=agent_constructor.__name__)

            preference_btn = wx.Button(scrolled_panel, label='...', size=(16, 16))
            # pref_button.SetBitmapLabel(wx.ArtProvider.GetBitmap(wx.ART_INFORMATION,
            #                                                     size=wx.Size(8, 8)))
            sizer.Add(preference_btn, 0, wx.ALIGN_CENTER)
            sizer.Add(checkbox, 0, wx.ALIGN_CENTER)

            items_sizer.Add(sizer, flag=wx.ALL, border=5)
            self.agent_checkedbuttons.append(checkbox)
            self.agent_preference_buttons.append(preference_btn)
            self.agent_preference_texts.append('')

            preference_btn.Bind(
                wx.EVT_BUTTON,
                # Need to wrap idx into a closure
                (lambda _: lambda e: self._open_preference_dialog(_))(int(idx)))

        scrolled_panel.SetSizerAndFit(items_sizer)
        scrolled_panel.SetupScrolling()

        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)
        boxsizer.Add(scrolled_panel)

        return boxsizer

    def _open_preference_dialog(self, idx):
        dlg = AgentPreferenceDialog(self.agent_preference_texts[idx], parent=self)
        ret_code = dlg.ShowModal()
        if ret_code == 0:
            self.agent_preference_texts[idx] = dlg.text
        dlg.Destroy()

    def _make_simulation_box(self):
        sb = wx.StaticBox(self, label='Simulation')

        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)
        gridsizer = wx.GridSizer(rows=6, cols=3, hgap=5, vgap=5)

        self.start_sim_button = wx.Button(self, label='Start')
        # self.start_sim_button.SetBackgroundColour(wx.TheColourDatabase.Find('FOREST GREEN'))
        # self.start_sim_button.SetForegroundColour(wx.TheColourDatabase.Find('WHITE'))
        self.stop_sim_button = wx.Button(self, label='Stop')
        # self.stop_sim_button.SetBackgroundColour(wx.TheColourDatabase.Find('MEDIUM VIOLET RED'))
        # self.stop_sim_button.SetForegroundColour(wx.TheColourDatabase.Find('WHITE'))

        self.set_start_pos_button = wx.Button(self, label='Set start')
        self.start_pos_text = wx.TextCtrl(self)

        self.set_goal_pos_button = wx.Button(self, label='Set goal')
        self.goal_pos_text = wx.TextCtrl(self)
        self.goal_destination_mode = wx.CheckBox(self, label='dest mode')
        self.goal_destination_mode.SetValue(False)

        self.heading_text = wx.TextCtrl(self)
        self.max_steps_text = wx.TextCtrl(self)
        self.step_size_text = wx.TextCtrl(self)

        gridsizer.AddMany([
            (self.start_sim_button, 0, wx.EXPAND),
            (self.stop_sim_button, 0, wx.EXPAND),
            ((0, 0), 0, wx.EXPAND),
            (self.set_start_pos_button, 0, wx.EXPAND),
            (self.start_pos_text, 0, wx.EXPAND),
            ((0, 0), 0, wx.EXPAND),
            (self.set_goal_pos_button, 0, wx.EXPAND),
            (self.goal_pos_text, 0, wx.EXPAND),
            (self.goal_destination_mode, 0, wx.EXPAND),
            (wx.StaticText(self, label='Heading (deg)'), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL),
            (self.heading_text, 0, wx.EXPAND),
            ((0, 0), 0, wx.EXPAND),
            (wx.StaticText(self, label='Max steps'), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL),
            (self.max_steps_text, 0, wx.EXPAND),
            ((0, 0), 0, wx.EXPAND),
            (wx.StaticText(self, label='Step size'), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL),
            (self.step_size_text, 0, wx.EXPAND),
            ((0, 0), 0, wx.EXPAND),
        ])

        boxsizer.Add(gridsizer, flag=wx.ALL, border=5)

        return boxsizer

    def _make_options_box(self):
        sb = wx.StaticBox(self, label='Options')
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        self.no_goal = wx.CheckBox(self, label='No goal')
        boxsizer.Add(self.no_goal, flag=wx.ALL, border=5)

        self.no_planning = wx.CheckBox(self, label='No planning')
        boxsizer.Add(self.no_planning, flag=wx.ALL, border=5)

        self.replan = wx.CheckBox(self, label='Replan')
        boxsizer.Add(self.replan, flag=wx.ALL, border=5)

        self.stop_on_collision = wx.CheckBox(self, label='Stop on collision')
        boxsizer.Add(self.stop_on_collision, flag=wx.ALL, border=5)

        self.show_title = wx.CheckBox(self, label='Show title')
        boxsizer.Add(self.show_title, flag=wx.ALL, border=5)

        self.show_legends = wx.CheckBox(self, label='Show legends')
        boxsizer.Add(self.show_legends, flag=wx.ALL, border=5)

        self.show_axes_ticks = wx.CheckBox(self, label='Show axes ticks')
        boxsizer.Add(self.show_axes_ticks, flag=wx.ALL, border=5)

        self.show_waypoints = wx.CheckBox(self, label='Show waypoints')
        boxsizer.Add(self.show_waypoints, flag=wx.ALL, border=5)

        self.show_obstacles = wx.CheckBox(self, label='Show laser scan')
        boxsizer.Add(self.show_obstacles, flag=wx.ALL, border=5)

        self.show_traj = wx.CheckBox(self, label='Show trajectories')
        boxsizer.Add(self.show_traj, flag=wx.ALL, border=5)

        self.show_control_point_accels = wx.CheckBox(self, label='Show control point accels')
        boxsizer.Add(self.show_control_point_accels, flag=wx.ALL, border=5)

        self.show_control_point_metrics = wx.CheckBox(self, label='Show control point metrics')
        boxsizer.Add(self.show_control_point_metrics, flag=wx.ALL, border=5)

        self.show_accel = wx.CheckBox(self, label='Show accel')
        boxsizer.Add(self.show_accel, flag=wx.ALL, border=5)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.save_screenshot = wx.CheckBox(self, label='Save screenshot')
        hbox.Add(self.save_screenshot, flag=wx.ALL, border=5)

        boxsizer.Add(hbox, border=5)

        return boxsizer


class AgentPreferenceDialog(wx.Dialog):
    def __init__(self, text, *args, **kwargs):
        super(AgentPreferenceDialog, self).__init__(*args, **kwargs)
        self.text = text
        self.text_ctl = None
        self._init_ui()
        self.SetTitle('Set Agent Parameters')

    def _init_ui(self):
        text_panel = wx.Panel(self)
        sb = wx.StaticBox(text_panel, label='Agent Parameters')
        sbs = wx.StaticBoxSizer(sb, orient=wx.VERTICAL)
        sbs.Add(wx.StaticText(text_panel,
                              label='format: key1=value1, key2=value2, ...\n'
                                    'They will be passed into agent constructor as kwargs'),
                              flag=wx.ALL, border=5)
        self.text_ctl = wx.TextCtrl(text_panel, style=wx.TE_MULTILINE)
        self.text_ctl.SetValue(self.text)
        sbs.Add(self.text_ctl, proportion=1, flag=wx.ALL | wx.EXPAND, border=5)

        text_panel.SetSizer(sbs)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        ok_button = wx.Button(self, label='OK')
        cancel_button = wx.Button(self, label='Cancel')
        hbox.Add(ok_button)
        hbox.Add(cancel_button, flag=wx.LEFT, border=5)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(text_panel, proportion=1, flag=wx.ALL | wx.EXPAND, border=5)
        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, border=10)

        self.SetSizer(vbox)

        ok_button.Bind(wx.EVT_BUTTON, self.on_ok)
        cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)

    def on_ok(self, e):
        self.text = self.text_ctl.GetValue()
        self.EndModal(0)

    def on_cancel(self, e):
        self.EndModal(1)


class StatusCell(wx.Panel):
    def __init__(self, name, parent, textwidth, color=(255, 0, 0), **kwargs):
        super(StatusCell, self).__init__(parent=parent, **kwargs)
        self.name = name
        self.info_font = wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.NORMAL)

        self.img_panel = wx.Panel(self)
        self.bitmap = wx.StaticBitmap(self.img_panel)
        self.last_img_size = None
        self.text_ctrls = {}

        color_rect_size = 8
        color_rect_border = 2
        info_sizer_border = 3

        name_font = wx.Font(8, wx.MODERN, wx.NORMAL, wx.BOLD)
        name_text = wx.StaticText(self, label=self._break_text(name_font, name, textwidth))
        name_text.SetFont(name_font)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        color_code = wx.Panel(self, size=(color_rect_size, -1))
        color_code.SetBackgroundColour(color)

        info_sizer = wx.BoxSizer(wx.VERTICAL)
        info_sizer.Add(name_text, 0, wx.EXPAND)
        info_sizer.Add(self.img_panel, 0, wx.CENTER | wx.TOP | wx.BOTTOM, border=3)
        self.info_sizer = info_sizer

        sizer.Add(color_code, 0, wx.EXPAND | wx.ALL, border=color_rect_border)
        sizer.Add(info_sizer, 1, wx.EXPAND | wx.ALL, border=info_sizer_border)
        self.SetSizer(sizer)

    def _break_text(self, font, text, width):
        dc = wx.ScreenDC()
        dc.SetFont(font)
        breaked_text = [text[0]]
        cursor = 0
        for i in range(1, len(text)):
            w, h = dc.GetTextExtent(text[cursor:i])
            if w > width:
                breaked_text.append('\n')
                cursor = i
            breaked_text.append(text[i])
        return ''.join(breaked_text)

    def set_value(self, key, value):
        if key not in self.text_ctrls:
            t = wx.StaticText(self, label='')
            t.SetFont(self.info_font)
            self.info_sizer.Add(t, 0, wx.ALL, border=0)
            self.text_ctrls[key] = t
            self.Layout()
            self.GetParent().Layout()
        self.text_ctrls[key].SetLabel('%s: %s' % (key, value))

    def set_image(self, img):
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)

        h, w = img.shape[:2]
        wxbmp = wx.BitmapFromBuffer(w, h, img)
        self.bitmap.SetBitmap(wxbmp)

        if self.last_img_size != (w, h):
            self.img_panel.SetClientSize(wx.Size(w, h))
            self.last_img_size = (w, h)
            self.Layout()
            self.GetParent().Layout()

        self.Update()


class StatusWindow(wx.Panel):
    def __init__(self, parent, textwidth, **kwargs):
        super(StatusWindow, self).__init__(parent=parent, **kwargs)
        self.boxsizer = wx.BoxSizer(wx.VERTICAL)
        self.agent_status_cells = {}
        self.SetSizer(self.boxsizer)
        self.textwidth = textwidth

    def reset(self, agent_names, agent_colors):
        self.boxsizer.Clear(delete_windows=True)
        self.agent_status_cells = {}

        assert len(agent_names) == len(agent_colors)

        for idx, (name, color) in enumerate(zip(agent_names, agent_colors)):
            cell = StatusCell(name, self, self.textwidth, color)
            cell.SetBackgroundColour('#dddddd')
            self.boxsizer.Add(cell, 0, wx.ALL | wx.EXPAND, border=3)
            self.agent_status_cells[name] = cell

    def set_text(self, name, key, text):
        self.agent_status_cells[name].set_value(key, text)

    def set_image(self, name, img):
        self.agent_status_cells[name].set_image(img)


class AgentLegendHandler(object):
    def __init__(self, color):
        self.color = color

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = Rectangle([x0, y0], width, height, facecolor=self.color,
                          transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


class SimGui(wx.Frame):
    StateNone, StateSetStartPos, StateSetGoalPos = range(3)
    SimStateStopped, SimStateRunning, SimStatePaused = range(3)

    def __init__(self, size=(1800, 1200)):
        super(SimGui, self).__init__(None, size=size)
        self.plotter = Plot(self)

        self.map = None
        self.map_visualizer = None

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.control_panel = Controls(self)
        self.status_window = StatusWindow(self, textwidth=256)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.record_slider = wx.Slider(
            self, value=0, minValue=0, maxValue=0, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.record_slider.Bind(wx.EVT_SLIDER, self.on_record_slider_scroll)
        self.record_slider.Disable()

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.plotter, 1, wx.EXPAND)
        vbox.Add(self.record_slider, 0, wx.ALL | wx.EXPAND, border=5)
        self.left_vbox = vbox

        hbox.Add(self.status_window, 0, wx.EXPAND)
        hbox.Add(vbox, 1, wx.EXPAND)
        hbox.Add(self.control_panel)

        self.SetSizer(hbox)

        self.control_panel.start_sim_button.Bind(wx.EVT_BUTTON, self.cmd_simulate)
        self.control_panel.stop_sim_button.Bind(wx.EVT_BUTTON, self.cmd_stop_simulation)
        self.control_panel.stop_sim_button.Disable()

        self.control_panel.set_start_pos_button.Bind(
            wx.EVT_BUTTON, lambda e: self.set_state(SimGui.StateSetStartPos))
        self.control_panel.set_goal_pos_button.Bind(
            wx.EVT_BUTTON, lambda e: self.set_state(SimGui.StateSetGoalPos))

        self.control_panel.heading_text.SetValue('0.0')
        self.control_panel.max_steps_text.SetValue('2000')
        self.control_panel.step_size_text.SetValue('0.1')

        self.control_panel.show_waypoints.SetValue(True)
        self.control_panel.show_waypoints.Bind(
            wx.EVT_CHECKBOX, lambda e: self.cmd_show_waypoints())

        self.plotter.canvas.mpl_connect('button_press_event', lambda ev: self.on_click_canvas(ev))

        self.state = None

        # Simulation related
        self.sim_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.sim_timer)

        self.sim_steps = 0
        self.sim_state = SimGui.SimStateStopped
        self.sim_max_steps = 0
        self.last_start_pos = None
        self.last_goal_pos = None
        self.waypoints = []
        self.last_waypoints = None
        self.destination = None
        self.reset_simulation()
        self.sim_thread = None
        self.agents = {}

        # graph elements
        self.cm = matplotlib.cm.get_cmap('Dark2')
        self.h_start_point = None
        self.h_goal_point = None
        self.h_waypoints = None
        self.h_legend = None

        self.h_background_rect = None  # A white patch to cover the whole area
        self.background = None
        self.background_bbox = None
        self.background_ax_limit = None

    def load_map(self, map):
        self.map = map
        self.map_visualizer = FindMapVisualizer(self.map)(self.map, self.plotter.ax)
        self.plotter.ax.set_title(map.__repr__(), fontsize=16)
        self.plotter.canvas.draw()
        self.map_visualizer.draw_map()

    def reset_simulation(self):
        self.waypoints = []
        self.destination = None
        self.sim_steps = 0

    def set_state(self, state):
        self.state = state

    def draw_start_pos(self, x=None, y=None):
        if x is None or y is None:
            x, y = self.get_start_pos()

        if self.h_start_point:
            self.h_start_point.set_xdata(x)
            self.h_start_point.set_ydata(y)
        else:
            self.h_start_point, = self.plotter.ax.plot(
                x, y, 'o', markeredgecolor=(0.8, 0, 0), markerfacecolor='none')
        self.plotter.canvas.draw()

    def draw_goal_pos(self, x=None, y=None):
        if x is None or y is None:
            x, y = self.get_goal_pos()
        if self.h_goal_point:
            self.h_goal_point.set_xdata(x)
            self.h_goal_point.set_ydata(y)
        else:
            self.h_goal_point, = self.plotter.ax.plot(x, y, '*', color=(0.8, 0, 0))
        self.plotter.canvas.draw()

    def get_start_pos(self):
        toks = self.control_panel.start_pos_text.GetValue().split(',')
        return [float(t) for t in toks]

    def get_goal_pos(self):
        toks = self.control_panel.goal_pos_text.GetValue().split(',')
        return [float(t) for t in toks]

    def set_start_pos(self, x, y):
        self.control_panel.start_pos_text.SetValue('%.2f, %.2f' % (x, y))
        self.draw_start_pos(x, y)

    def set_goal_pos(self, x, y):
        self.control_panel.goal_pos_text.SetValue('%.2f, %.2f' % (x, y))
        self.draw_goal_pos(x, y)

    def set_heading(self, heading):
        self.control_panel.heading_text.SetValue('%f' % heading)

    def draw_waypoints(self, xs, ys):
        if self.h_waypoints:
            self.h_waypoints.set_xdata(xs)
            self.h_waypoints.set_ydata(ys)
            return
        self.h_waypoints, = self.plotter.ax.plot(xs, ys, color='g', alpha=0.2)

    def set_sim_state(self, state):
        if state == SimGui.SimStateRunning:
            self.control_panel.start_sim_button.SetLabel('Pause')
            self.control_panel.stop_sim_button.Enable()
        elif state == SimGui.SimStatePaused:
            self.control_panel.start_sim_button.SetLabel('Resume')
            self.control_panel.stop_sim_button.Enable()
        elif state == SimGui.SimStateStopped:
            self.control_panel.start_sim_button.SetLabel('Start')
            self.control_panel.stop_sim_button.Disable()
        self.sim_state = state

    def cmd_stop_simulation(self, e):
        self.stop()
        print('simulation stopped')

    def _finish_current_step(self):
        if self.sim_thread is not None:
            self.sim_thread.join()
            self.sim_thread = None
            start_time = time.time()
            self.simulate_render(self.sim_steps)
            print('render time: %.3f' % (time.time() - start_time))
            self.sim_steps += 1

    def on_timer(self, e):
        if self.sim_thread is not None:
            if self.sim_thread.is_alive():
                return

            self.sim_thread = None

            start_time = time.time()
            self.simulate_render(self.sim_steps)
            print('render time: %.3f' % (time.time() - start_time))

            if self.control_panel.save_screenshot.GetValue():
                buf = self.make_screenshot()
                cv2.imwrite('sim_images/%05d.png' % self.sim_steps, buf)

                for name in self.agents:
                    agent = self.agents[name]['agent']
                    if hasattr(agent, 'img'):
                        img = self._prepare_img(agent.img, 256, 0.1)
                        cv2.imwrite('sim_images/%s_%05d.png' % (name, self.sim_steps), img)

            if self.sim_steps == 0:
                # Call Layout() after rendering the first frame because the plot view may change
                # its size.
                self.Layout()

            self.sim_steps += 1

            n_sim_completed = 0
            for name in self.agents:
                agent = self.agents[name]['agent']
                if agent.stopped():
                    print('agent %s stopped' % name)
                    n_sim_completed += 1
                elif agent.reached_goal():
                    print('agent %s reached goal' % name)
                    n_sim_completed += 1

            if n_sim_completed == len(self.agents):
                print('simulation completes')
                self.stop()
                return

            if self.sim_steps >= self.sim_max_steps:
                print('max steps reached')
                self.stop()
                return

        self.sim_thread = threading.Thread(target=lambda: self.simulate_step())
        self.sim_thread.start()

    def on_record_slider_scroll(self, e):
        step = self.record_slider.GetValue()

        for name in self.agents:
            history = self.agents[name]['history']
            state = history[min(step, len(history) - 1)]
            self.agents[name]['agent'].restore_state(state)

        self.simulate_render(step)

    def pause(self):
        self.sim_timer.Stop()
        self._finish_current_step()
        self.set_sim_state(SimGui.SimStatePaused)
        self.record_slider.Enable()
        self.record_slider.SetMin(0)
        self.record_slider.SetMax(self.sim_steps - 1)
        self.record_slider.SetValue(self.sim_steps - 1)

    def resume(self):
        self.set_sim_state(SimGui.SimStateRunning)
        self.record_slider.Disable()

        for name in self.agents:
            history = self.agents[name]['history']
            state = history[-1]
            self.agents[name]['agent'].restore_state(state)

        self.sim_timer.Start(1, oneShot=wx.TIMER_CONTINUOUS)

    def stop(self):
        if self.sim_state == SimGui.SimStateRunning:
            self.pause()
        self.reset_simulation()
        self.sim_timer.Stop()
        self.set_sim_state(SimGui.SimStateStopped)
        self.record_slider.Disable()

    def cmd_simulate(self, e):
        if self.sim_state == SimGui.SimStateRunning:
            self.pause()
            return
        elif self.sim_state == SimGui.SimStatePaused:
            self.resume()
            return

        self.reset_simulation()

        param_utils.global_params.reload()

        start_pos = np.array(self.get_start_pos())
        goal_pos = np.array(self.get_goal_pos())

        if self.control_panel.no_goal.GetValue():
            waypoints = []
        elif self.control_panel.no_planning.GetValue():
            waypoints = [goal_pos]
        else:
            if self.last_start_pos is not None and self.last_goal_pos is not None \
                    and np.all(self.last_start_pos == start_pos) and np.all(self.last_goal_pos == goal_pos):
                print('reuse waypoints')
                waypoints = self.last_waypoints
            else:
                if self.control_panel.goal_destination_mode.GetValue():
                    dest = tuple(self.map.get_destination_from_map_coord(goal_pos[0], goal_pos[1]))
                    waypoints = self.map.find_path_destination(start_pos, dest)
                else:
                    waypoints = self.map.find_path(start_pos, goal_pos)

                self.last_waypoints = waypoints
                self.last_start_pos = start_pos
                self.last_goal_pos = goal_pos

        if waypoints is None:
            print('infeasible')
            return

        self.init_agents()

        if self.control_panel.replan.GetValue():
            # TODO: remove code duplication
            dest = tuple(self.map.get_destination_from_map_coord(goal_pos[0], goal_pos[1]))
            if dest not in self.map.reachable_locs_per_destination:
                print('invalid destination')
            self.destination = dest

        for name in self.agents:
            agent = self.agents[name]['agent']
            agent.reset()
            agent.set_map(self.map)
            agent.set_pos(start_pos)
            agent.set_heading(np.deg2rad(float(self.control_panel.heading_text.GetValue())))
            agent.set_waypoints(waypoints)

            if hasattr(agent, 'set_destination'):
                # This agent uses destination as its goal
                dest = tuple(self.map.get_destination_from_map_coord(goal_pos[0], goal_pos[1]))
                if dest not in self.map.reachable_locs_per_destination:
                    print('invalid destination')
                else:
                    agent.set_destination(dest)
                print('destination label', dest)

        if waypoints is not None and len(waypoints) > 0:
            self.draw_waypoints(*zip(*waypoints))
        self.waypoints = waypoints

        self.draw_start_pos()
        self.draw_goal_pos()

        show_ticks = self.control_panel.show_axes_ticks.GetValue()
        self.plotter.ax.get_xaxis().set_visible(show_ticks)
        self.plotter.ax.get_yaxis().set_visible(show_ticks)

        if self.control_panel.show_title.GetValue():
            self.plotter.ax.set_title(self.map.__repr__(), fontsize=16)
        else:
            self.plotter.ax.set_title('')

        agent_names = sorted(list(self.agents.keys()))

        def to_uint8(color):
            return int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)

        agent_colors = [to_uint8(self.agents[_]['color']) for _ in agent_names]
        self.status_window.reset(agent_names, agent_colors)

        self.sim_max_steps = int(self.control_panel.max_steps_text.GetValue())

        self.sim_timer.Start(1, oneShot=wx.TIMER_CONTINUOUS)
        self.set_sim_state(SimGui.SimStateRunning)

    def _get_color(self, idx):
        return self.cm(idx % self.cm.N)

    def init_agents(self):
        # Clear current agents' visualization items.
        for _, agent in iteritems(self.agents):
            agent['visualizer'].clear()

        agent_switches = [b.GetValue() for b in self.control_panel.agent_checkedbuttons]

        new_agents = {}

        agent_idx = 0
        for i in range(len(agent_switches)):
            if not agent_switches[i]:
                continue

            agent_constructor = agent_factory.public_agents[i]
            name = agent_constructor.__name__

            # Reuse agent object if possible because some agents
            # take a long time to initialize.
            # FIXME: HACK!
            # We don't reuse non-visual agents because we might change their parameters
            # if name in self.agents and hasattr(self.agents[name]['agent'], 'img'):
            #     instance = self.agents[name]['agent']
            # else:

            agent_kwargs = str_to_dict(self.control_panel.agent_preference_texts[i])
            instance = agent_constructor(**agent_kwargs)

            c = self._get_color(agent_idx)
            vis = FindAgentVisualizer(instance)(
                self.plotter.ax,
                draw_control_point=False,
                traj_color=c,
                obstacle_color=c,
                heading_color=c,
                active_wp_color=c,
                goal_color=c,
                label=name
            )

            new_agents[name] = {
                'agent': instance,
                'color': c,
                'visualizer': vis,
                'legend_handler': AgentLegendHandler(c),
                'history': [],
                'sim_state': {'traj': [], 'collisions': 0}
            }

            agent_idx += 1

        # Delete agents that are no longer used
        for name, agent in iteritems(self.agents):
            if name not in new_agents:
                del agent['agent']

        self.agents = new_agents

    def draw_background(self, force_redraw=False):
        '''
        :return: draw background and cache it.
        '''
        # Pan/zoom can change axis limits. In that case we need to redraw the background.
        x1, x2 = self.plotter.ax.get_xlim()
        y1, y2 = self.plotter.ax.get_ylim()

        def draw_legends():
            agent_ids = self.agents.keys()
            if self.control_panel.show_legends.GetValue():
                self.h_legend = self.plotter.ax.legend(
                    agent_ids, agent_ids,
                    handler_map={
                        agent_id: self.agents[agent_id]['legend_handler'] for agent_id in agent_ids},
                    loc='lower left', bbox_to_anchor=(0, -0.1), ncol=5)
            elif self.h_legend:
                self.h_legend.remove()
                self.h_legend = None

        if self.background is None or \
           self.background_bbox != self.plotter.ax.bbox.__repr__() or \
           self.background_ax_limit != (x1, x2, y1, y2) or\
           force_redraw:

            self.plotter.ax.autoscale(False)

            draw_legends()

            # Cover all graphical elements
            if self.h_background_rect is None:
                self.h_background_rect = Rectangle((x1, y1), x2 - x1, y2 - y1, color='w')
                self.h_background_rect.set_zorder(10**5)
                self.plotter.ax.add_patch(self.h_background_rect)
            else:
                self.h_background_rect.set_bounds(x1, y1, x2 - x1, y2 - y1)
                self.h_background_rect.set_visible(True)
                self.plotter.ax.draw_artist(self.h_background_rect)

            self.plotter.canvas.draw()

            # Draw the map
            self.map_visualizer.draw_map()

            self.h_background_rect.set_visible(False)

            self.plotter.canvas.blit(self.plotter.ax.bbox)

            self.background = self.plotter.canvas.copy_from_bbox(self.plotter.ax.bbox)
            self.background_bbox = self.plotter.ax.bbox.__repr__()

            # limits might get changed. Retrieve new limits here.
            x1, x2 = self.plotter.ax.get_xlim()
            y1, y2 = self.plotter.ax.get_ylim()
            self.background_ax_limit = (x1, x2, y1, y2)
        else:
            self.plotter.canvas.restore_region(self.background)

    def simulate_step(self):
        # Note that this runs in another thread.
        step_size = float(self.control_panel.step_size_text.GetValue())

        print('step %d step_size: %.3f' % (self.sim_steps, step_size))

        # Update agent states
        for name in self.agents:
            start_time = time.time()

            agent = self.agents[name]['agent']
            if agent.collide() and self.control_panel.stop_on_collision.GetValue():
                pass
            else:
                if self.control_panel.replan.GetValue():
                    waypoints = self.map.find_path_destination(agent.pos, self.destination)
                    if waypoints is None or len(waypoints) == 0:
                        print('no valid waypoint. Use previous.')
                    else:
                        agent.set_waypoints(waypoints)
                        agent.wp_idx = 0

                agent.step(step_size)
                if agent.collide():
                    self.agents[name]['sim_state']['collisions'] += 1

                self.agents[name]['history'].append(agent.save_state())

                print('agent %s step time: %.3f '
                      'pos %.3f %.3f heading %.3f deg vel_norm %.3f collisions: %d' % \
                      (name, time.time() - start_time,
                       agent.pos[0], agent.pos[1],
                       np.rad2deg(agent.heading),
                       np.linalg.norm(agent.velocity),
                       self.agents[name]['sim_state']['collisions']))

            pos = agent.pos
            self.agents[name]['sim_state']['traj'].append([pos[0], pos[1]])

    def simulate_render(self, step_idx):
        # Always redraw at step 0
        self.draw_background(force_redraw=(step_idx == 0))

        self.plotter.ax.draw_artist(self.h_start_point)
        self.plotter.ax.draw_artist(self.h_goal_point)

        if self.h_waypoints is not None:
            self.plotter.ax.draw_artist(self.h_waypoints)

        for name in self.agents:
            v = self.agents[name]
            agent = v['agent']
            sim_state = v['sim_state']
            visualizer = v['visualizer']

            visualizer.draw_trajectory(*zip(*sim_state['traj'][:step_idx+1]))
            visualizer.draw_agent_state(agent)

            visualizer.set_show_trajectory(self.control_panel.show_traj.GetValue())
            visualizer.set_show_accel(self.control_panel.show_accel.GetValue())
            visualizer.set_show_control_point_accels(self.control_panel.show_control_point_accels.GetValue())
            visualizer.set_show_control_point_metrics(self.control_panel.show_control_point_metrics.GetValue())
            visualizer.set_show_obstacles(self.control_panel.show_obstacles.GetValue())

            if hasattr(agent, 'img') and agent.img is not None:
                img = (np.array(agent.img.transpose(1, 2, 0), copy=True) * 255.0).astype(np.uint8)
                self._draw_all_waypoints(img, agent)
                self.status_window.set_image(name, img)

            self.status_window.set_text(name, 'velocity', '%.3f m/s' % np.linalg.norm(agent.velocity))
            self.status_window.set_text(name, 'angular_vel', '%.1f deg/s' % np.rad2deg(agent.angular_velocity))

        self.plotter.canvas.blit(self.plotter.ax.bbox)

    def _prepare_img(self, img, img_width, brightness_adj):
        img = np.transpose(img, (1, 2, 0))
        height = int(float(img_width) / img.shape[1] * img.shape[0])
        img = cv2.resize(img, (img_width, height))
        if img.dtype == np.float32:
            img = np.clip(((img + brightness_adj) * 255), 0.0, 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _draw_all_waypoints(self, img, agent):
        """
        Draw the groundtruth and neural waypoints if available
        TODO: remove hardcoded colors.
        :param img: 3-channel uint8 image
        """
        if len(agent.goals_global) > 0:
            self._draw_waypoint(
                img, agent, agent.global_to_local(agent.goals_global[0]), (0, 255, 0))

    def _draw_waypoint(self, img, agent, wp_local, color):
        if hasattr(agent, 'camera_pos'):
            cam_pos = agent.camera_pos
        else:
            return

        # Select a sensor that has 'h_fov' attribute.
        for sensor in agent.sensors:
            if hasattr(sensor, 'h_fov'):
                fov = sensor.h_fov

        img_width = img.shape[1]

        def to_image_space(p):
            size = max(int(10.0 / (np.linalg.norm(p) + 0.01)), 5)
            l = img_width * 0.5 / np.tan(0.5 * fov)
            d = p[1] / p[0] * l
            return d, size

        wp_local[0] -= cam_pos[0]
        wp_local[1] -= cam_pos[1]

        if wp_local[0] < 0:
            # Waypoint behind the agent. Ignore.
            return

        d, s = to_image_space(wp_local)
        cv2.circle(img, (img.shape[1] // 2 - int(d), img.shape[0] // 2), s, color, 2)

    def make_screenshot(self):
        w, h = self.plotter.canvas.get_width_height()
        buf = np.fromstring(self.plotter.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

        # If the agent uses visual images, also save them.
        img_width = 512
        brightness_adj = 0.1
        strip = np.ones((h, img_width, 3), np.uint8) * 128
        y = 0

        for name in self.agents:
            agent = self.agents[name]['agent']
            if hasattr(agent, 'img'):
                img = self._prepare_img(agent.img, img_width, brightness_adj)
                self._draw_all_waypoints(img, agent)

                if hasattr(agent, 'img_left'):
                    img_left = self._prepare_img(agent.img_left, img_width, brightness_adj)
                    img = np.concatenate([img_left, img], axis=1)

                if hasattr(agent, 'img_right'):
                    img_right = self._prepare_img(agent.img_right, img_width, brightness_adj)
                    img = np.concatenate([img, img_right], axis=1)

                label_banner_cairo = np.zeros((32, img.shape[1]), np.uint32)

                surface = cairo.ImageSurface.create_for_data(
                    label_banner_cairo, cairo.FORMAT_ARGB32,
                    label_banner_cairo.shape[1], label_banner_cairo.shape[0])
                cr = cairo.Context(surface)

                if agent.collide():
                    cr.set_source_rgb(255, 0, 0)
                else:
                    cr.set_source_rgb(255, 255, 255)

                cr.select_font_face('Helvetica', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                cr.set_font_size(24)
                cr.move_to(0, 24)
                cr.show_text(name)

                label_banner = cairo_argb_to_opencv_bgr(label_banner_cairo)

                img = np.concatenate([label_banner, img], axis=0)

                if y + img.shape[0] > strip.shape[0]:
                    # Create new if current is full
                    buf = np.concatenate([buf, strip], axis=1)
                    strip = np.zeros((h, img_width, 3), np.uint8)
                    y = 0

                # Enlarge the strip's width if img's width is larger than the strip's width
                if strip.shape[1] < img.shape[1]:
                    padding = np.zeros(
                        (strip.shape[0], img.shape[1] - strip.shape[1], strip.shape[2]),
                        dtype=strip.dtype)
                    strip = np.concatenate([strip, padding], axis=1)

                strip[y: y + img.shape[0], 0: img.shape[1], :] = img
                y += img.shape[0]

        if y > 0:
            buf = np.concatenate([buf, strip], axis=1)
        return buf

    def on_click_canvas(self, event):
        if self.state == SimGui.StateSetStartPos:
            self.set_start_pos(event.xdata, event.ydata)
            self.state = SimGui.StateNone
        elif self.state == SimGui.StateSetGoalPos:
            self.set_goal_pos(event.xdata, event.ydata)
            self.state = SimGui.StateNone

    def on_close(self, e):
        self.sim_timer.Stop()
        del self.agents
        e.Skip()

    def cmd_show_waypoints(self):
        if self.h_waypoints:
            self.h_waypoints.set_visible(self.control_panel.show_waypoints.GetValue())
