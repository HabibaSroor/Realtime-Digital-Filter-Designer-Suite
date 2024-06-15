from PyQt6.QtGui import QEnterEvent, QDoubleValidator
from PyQt6.QtGui import QIntValidator
from ui import Ui_MainWindow
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QGridLayout, QMenu
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, \
    QRadioButton, QSpacerItem, QSlider, QFileDialog, QDialog, QListWidgetItem
import pyqtgraph as pg
import numpy as np
from scipy import signal
from PyQt6.QtCore import QEvent, Qt, QTimer
import pandas as pd
from all_pass import Ui_Dialog
from functools import partial
from PyQt6.QtCore import QPointF

    
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.zeroes_coords = []
        self.poles_coords = []
        self.initUI()

        self.addPtsToGraph()
        self.tab = 0

        self.coords = [self.zeroes_coords, self.poles_coords] #nearly not used

        self.padding_timer = QTimer()
        self.padding_timer.setInterval(25)
        self.padding_timer.timeout.connect(self.draw_signal)

        self.time = np.arange(0, 10001, 1)
        self.ui.horizontalSlider.setValue(40)

        self.ui.pad_widget = CustomPaddingWidget()
        self.ui.tabWidget_2.addTab(self.ui.pad_widget, "")
        self.ui.tabWidget_2.setTabText(self.ui.tabWidget_2.indexOf(self.ui.pad_widget), self.tr("Padding Area"))

        self.ui.pad_widget.setMouseTracking(True)
        self.ui.zeroes_poles_graph.setMenuEnabled(False)
        self.filtered_sig = []
        self.sig = None

        self.zeros_to_main = []
        self.poles_to_main = []

        self.data = []
        self.a_values_list = []
        self.all_pass_transfer_function = None
        self.signal_plotWidgets = [self.ui.signal_input_graph, self.ui.filter_output_graph]
        # self.ui.filter_output_graph.plotItem.setYLink(self.ui.signal_input_graph.plotItem)
        self.ui.listWidget.itemDoubleClicked.connect(self.deactivate_allpass)
        self.ui.filter_output_graph.setXLink(self.ui.signal_input_graph)
    
    def deactivate_allpass(self, item):

        a_value = item.text()

        if "deactivated" in a_value:
            a_value = a_value.split(" ")[0] 
            item.setText(a_value)
            pole = complex(a_value)
            pole_list = [pole.real, pole.imag]
            zero = 1 / pole.conjugate()
            zero_list = [zero.real, zero.imag]
            self.zeroes_coords.append(zero_list)
            self.poles_coords.append(pole_list)
            
        else:

            pole = complex(a_value)
            pole_list = [pole.real, pole.imag]
            zero = 1 / pole.conjugate()
            zero_list = [zero.real, zero.imag]
            self.zeroes_coords.remove(zero_list)
            self.poles_coords.remove(pole_list)
            item.setText(a_value + " (deactivated)")
        
        self.addPtsToGraph()
        self.setup_filter_from_zeroes_poles()
        self.plot_responses()


    def initUI(self):
        # Creating a custom Graph Item to control the points on the unit circle space representing the zeors and poles 
        self.graph_item = CustomGraphItem()
        # zeroes_poles_graph is a promoted plotwidget, representing the  add the custom graph item to it
        self.ui.zeroes_poles_graph.addItem(self.graph_item)
        unit_circ = pg.PlotDataItem(x=np.cos(np.linspace(0, 2 * np.pi, 360)), y=np.sin(np.linspace(0, 2 * np.pi, 360)))
        self.ui.zeroes_poles_graph.addItem(unit_circ)
        #print(" self.ui.zeroes_poles_graph ",self.ui.zeroes_poles_graph)
        self.ui.zeroes_poles_graph.setAspectLocked()
        self.ui.zeroes_poles_graph.setRange(xRange=(-2, 2), yRange=(-2, 2))
        
        # Add GridItem for the grid and infinite lines as axis
        x_axis_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w'))
        y_axis_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w'))
        self.ui.zeroes_poles_graph.addItem(x_axis_line)
        self.ui.zeroes_poles_graph.addItem(y_axis_line)
        self.grid = pg.GridItem()
        # Disable numbering on the grid lines 
        self.grid.setTextPen(None)
        self.ui.zeroes_poles_graph.addItem(self.grid)
        # mag_response_garph is a promoted plotwidget, representing the magnitue response of the designed filter
        # add the unit circle, the grid and the axis to it
        # Specify the labels on the left and button of the axis
        self.ui.mag_response_garph.setLabel('left', "Magnitude Response (dB)")
        self.ui.mag_response_garph.setLabel('bottom', "w normalized")
        # phase_response_garph is a promoted plotwidget, representing the phase response of the designed filter
        # add the unit circle, the grid and the axis to it
        # Specify the labels on the left and button of the axis
        self.ui.phase_response_garph.setLabel('left', "Phase Response")
        self.ui.phase_response_garph.setLabel('bottom', "w normalized")
        self.ui.phase_response_garph.setXLink("mag_plotitem")
        # Initializing the current loaded signal as None
        self.current_orig_sig = None
        # Initializing the timer for real life updating
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateSignals)
        self.timer.setInterval(100)
        self.count = 0

        self.ui.pushButton_clear_zeros.clicked.connect(partial(self.onClearXClicked, 0))
        self.ui.pushButton_clear_poles.clicked.connect(partial(self.onClearXClicked, 1))
        self.ui.pushButton_clear_all.clicked.connect(self.onClearAllClicked)
        self.ui.pushButton_6.clicked.connect(self.load_signal)
        self.ui.tabWidget_2.currentChanged.connect(self.signalTypeChange)
        self.ui.horizontalSlider.sliderReleased.connect(self.changeResolution)
        self.ui.pushButton.clicked.connect(self.openDialog)
        self.ui.zeroes_poles_graph.scene().sigMouseClicked.connect(self.AddNewZeroPole)

        self.setWindowTitle('Filter Designer Suite')
        self.show() 

    

    def AddNewZeroPole(self, event):
        # Defining the right click of the mouse to add a zero or a pole (according to the combo box)
        if event.button() == Qt.MouseButton.RightButton:
            # Get the position of the click on the plotwidget
            pos = event.scenePos()
            # Map the mouse coordinates to the view coordinates of the GridItem

            view_coords = self.grid.mapFromScene(pos)
            # This flag returns a zero for the (zero) in the filter
            # and one for the (pole) in the filter
            zero_or_pole = self.ui.comboBox_pole_zero.currentText() == "Pole"
            # to_be_added tuple contains the coordintes of the point and whether it's a zero or a pole
            to_be_added = (view_coords, zero_or_pole)
            self.addZeroesPoles(to_be_added)
            # print(self.graph_item_left.allChildItems()[0].getData()) #-> this is how to get the coords

    def changeResolution(self):
        speed = int(self.ui.horizontalSlider.value())
        self.timer.setInterval(speed)

    def draw_signal(self):
        self.data.append(self.ui.pad_widget.amplitude)
        # print(f"amplitude {self.ui.pad_widget.amplitude}")
        if isinstance(self.filterSignal(self.ui.pad_widget.amplitude), int):
            self.filtered_pad_sig[self.count] = self.ui.pad_widget.amplitude
            # print(self.filtered_pad_sig[self.count])
        else:
            if len(self.filterSignal(self.ui.pad_widget.amplitude)) != 0:
                self.filtered_pad_sig[self.count - 1] = self.filterSignal(self.ui.pad_widget.amplitude)[self.count - 1]
        self.updateSignals()

    def signalTypeChange(self):
        # 0 is load, 1 is pad
        self.tab = not self.tab
        self.timer.stop()
        self.count = 0
        
        if self.tab: # padding
            # self.ui.filter_output_graph.plotItem.setYLink(None)
            # range for padding signal
            # self.ui.signal_input_graph.setYRange(min = 10, max = 60)
            self.ui.signal_input_graph.setXRange(min=-300, max=100)
            self.ui.filter_output_graph.enableAutoRange(axis=pg.ViewBox.YAxis)
            # self.padding_timer.start()
        else:
            pass
            # self.ui.filter_output_graph.plotItem.setYLink(self.ui.signal_input_graph.plotItem)

    def openDialog(self):
            zeros = MyMainWindow.to_complex(self.zeroes_coords)
            poles = MyMainWindow.to_complex(self.poles_coords)
            if self.tab:
                self.padding_timer.stop()
            else:
                self.timer.stop()
    
            # print ("zeros and poles in mainwindow ", zeros, poles)
            dialog = MyDialog(zeros, poles, self.a_values_list)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.a_values_list, self.all_pass_transfer_function = dialog.return_to_main_window()
                if (self.a_values_list): 
                    self.poles_to_main = [[p.real, p.imag]  for p in self.a_values_list]
                    self.zeros_to_main = [[(1/p.conjugate()).real, (1/p.conjugate()).imag]  for p in self.a_values_list]
                    print("Received values in main window:", self.poles_to_main," and the zeros ", self.zeros_to_main)
                
            self.update_after_phase_correction(self.zeros_to_main, self.poles_to_main)
            if self.tab:
                self.padding_timer.start()
            else:
                self.timer.start()

    def update_after_phase_correction(self, new_zeroes=[], new_poles=[]):
        self.zeroes_coords += new_zeroes
        self.poles_coords += new_poles

        for a_allpass in self.a_values_list:
            self.ui.listWidget.addItem(str(complex(a_allpass)))

        self.addPtsToGraph()
        self.setup_filter_from_zeroes_poles()
        self.plot_responses()
        # print(f"zero {self.zeroes_coords} poles {self.poles_coords}")
    
    def load_signal(self):

        path, format = QFileDialog.getOpenFileName(None, f"Load Image")
        self.count = 0

        signal_df = pd.read_csv(path)
        self.time = signal_df.iloc[:, 0]
        self.sig = signal_df.iloc[:, 1]

        self.filtered_sig = np.zeros(10000, dtype=complex)

        self.setup_graphs()
        #TODO: also call this function when an element is removed
        self.setup_filter_from_zeroes_poles()
        self.timer.start()

    def format_list_to_complex(self, list):
        op_list = []
        for root_tuple in MyMainWindow.to_complex(list):
            
            op_list.append(root_tuple)
        return op_list

    #TODO: maybe sth can be done for the repitition
    def setup_filter_from_zeroes_poles(self):
        if not self.check_if_filter_exist():
            return
        
        zeroes_list = self.format_list_to_complex(self.zeroes_coords)
        poles_list = self.format_list_to_complex(self.poles_coords)
        
        if len(zeroes_list) == 0:
            zeroes_list = np.zeros_like(poles_list)
        elif len(poles_list) == 0:
            poles_list = np.zeros_like(zeroes_list)

        # get coeffs of difference eqn [... x[2] x[1] x[0]]
        self.zeroes_x_coefficients = np.poly(zeroes_list)
        self.poles_y_coefficients = np.poly(poles_list)

        #flip
        self.poles_y_coefficients = self.poles_y_coefficients[::-1]
        self.zeroes_x_coefficients = self.zeroes_x_coefficients[::-1]
        
        # Initialize buffers
        self.buffer_x = np.zeros(len(self.zeroes_x_coefficients), dtype=complex)  # input buffer   
        self.buffer_y = np.zeros(len(self.poles_y_coefficients), dtype=complex)  # output buffer
    
    def setup_graphs(self):

        self.data = []
        self.filtered_pad_sig = np.zeros(10000, dtype=complex)
        self.ui.signal_input_graph.setYRange(min=np.min(self.sig), max=np.max(self.sig))
        self.ui.filter_output_graph.setYRange(min=np.min(self.sig), max=np.max(self.sig))
        self.ui.filter_output_graph.autoRange(True)
        # self.ui.filter_output_graph.setYRange(min=np.min(self.filtered_sig), max=np.max(self.filtered_sig))
        time_step = self.time[1]-self.time[0]
        for plot_widget in self.signal_plotWidgets:
            plot_widget.clear()
            plot_widget.setXRange(min=-45*time_step, max=45*time_step)

    # filter point by point
    def filterSignal(self, x_n):
        if not self.check_if_filter_exist():
            return -1
    
        # print(f"self.poles_y_coefficients {self.poles_y_coefficients}")
        # print(f"self.zero coeff {self.zeroes_x_coefficients}")
        
        #shift buffers
        self.buffer_x[:-1] = self.buffer_x[1:]
        self.buffer_y[:-1] = self.buffer_y[1:]

        # add current input to x_buffer
        self.buffer_x[-1] = x_n
        # print(f"self.buffer_x {self.buffer_x}")
        # print(f"self.ubuffer {self.buffer_y}")
        # Apply difference equation
        # print(f"self.buffer_x {self.buffer_x.size}")
        # print(f"self.buffer_y {self.buffer_y.size}")
        y = (np.sum(self.zeroes_x_coefficients * self.buffer_x) - np.sum(self.poles_y_coefficients[:-1] * self.buffer_y[:-1])) * (1 / self.poles_y_coefficients[-1]) # if y[n] has a coefficient, will not happen here
        # print(f"y {y}")
        # Store output sample
        self.buffer_y[-1] = y

        if self.tab:
            list = self.filtered_pad_sig
        else:
            list = self.filtered_sig
        #return the whole o/p list till now
        list[self.count] = y

        return list[:self.count]
    
    def updateSignals(self):
        if not self.tab: # load signal
            if self.sig is not None:
                if self.count == len(self.sig):
                    self.timer.stop()
                    return
                self.updateIndividualSignal(self.sig, self.ui.signal_input_graph)
                if self.check_if_filter_exist():
                    self.filtered_sig[:self.count] = self.filterSignal(self.sig[self.count])
                else:
                    self.filtered_sig[self.count] = self.sig[self.count]
                self.updateIndividualSignal(self.filtered_sig, self.ui.filter_output_graph)
        else:

            self.updateIndividualSignal(self.filtered_pad_sig, self.ui.filter_output_graph)
            self.updateIndividualSignal(self.data, self.ui.signal_input_graph)
        self.count += 1

    def updateIndividualSignal(self, signal_values, plot_widget):
        plot_widget.plot(self.time[:self.count], np.abs(signal_values[:self.count]))
        plot_widget.getViewBox().translateBy(x=(self.time[1] - self.time[0])/2)

    def check_if_filter_exist(self):
        return len(self.zeroes_coords) + len(self.poles_coords) != 0
    
    @staticmethod
    def find_closest_point(target_coord, coord_list):
        target_coord = np.array(target_coord)
        coord_list = np.array(coord_list)

        # Calculate Euclidean distances
        distances = np.linalg.norm(coord_list - target_coord, axis=1)
        # if np.min(distances) > 1:
        #     return ([], False, -1)

        # Find the index of the minimum distance
        closest_index = np.argmin(distances)

        # Return the closest point and its index
        closest_point = coord_list[closest_index]
        # return (closest_point, True, closest_index)
        return closest_point

    def addPtsToGraph(self):
        # Define positions of nodes
        self.graph_item.scatter.clear()
        # print(self.zeroes_coords)

        self.pos_arr = np.array(self.zeroes_coords + self.poles_coords)
        
        if len(self.pos_arr) == 0:
            self.graph_item.setData(pos=np.empty((0, 2), dtype=float)
                                    )
        else:
            symbols_arr = ['o'] * len(self.zeroes_coords) + ['x'] * len(self.poles_coords)

            self.graph_item.setData(pos=self.pos_arr,
                                    size=.1, symbol=symbols_arr, pxMode=False,
                                    )

    def plot_responses(self):
        self.ui.mag_response_garph.clear()
        self.ui.phase_response_garph.clear()

        complex_zeroes = self.to_complex(self.zeroes_coords)
        complex_poles = self.to_complex(self.poles_coords)
        
        omegas, freq_response = signal.freqz_zpk(complex_zeroes, complex_poles, 1)

        h_mag = 20 * np.log10(abs(freq_response))
        h_phase = np.unwrap(np.angle(freq_response))
        omegas = np.around(omegas, 3)

        self.filterOrder = max(len(self.zeroes_coords), len(self.poles_coords))
        self.ui.mag_response_garph.plot(omegas, h_mag)
        self.ui.phase_response_garph.plot(omegas, h_phase)

    def addZeroesPoles(self, to_be_added):
        
        real = to_be_added[0].x()
        img = to_be_added[0].y()

        #to avoid division by zero
        real += 1e-10
        img += 1e-10

        # zero -> zero, one -> pole
        list = self.coords[to_be_added[1]]
        list.append([real, img]) 

        if self.ui.radioButton_conj.isChecked():
            list.append([real, -1 * img])

        self.addPtsToGraph()
        #TODO: also call this function when an element is removed
        self.setup_filter_from_zeroes_poles()
        self.plot_responses()

    def onClearXClicked(self, index_to_clear):
        self.coords[index_to_clear].clear()
        main_window_instance.addPtsToGraph()
        self.plot_responses()

    def onClearAllClicked(self):
        self.onClearXClicked(0)
        self.onClearXClicked(1)

    @staticmethod
    def to_complex(list):
        # list must be a list of lists
        complex_list = []
        for element in list:
            complex_list.append(element[0] + 1.0j * element[1])
        return complex_list
    

class MyDialog(QDialog, Ui_Dialog):
    def __init__(self):
        super(MyDialog, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        

    def __init__(self, complex_zeroes_original, complex_poles_original, coming_a_values):
        super(MyDialog, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.complex_zeroes_original=complex_zeroes_original
        self.complex_poles_original=complex_poles_original
        self.coming_a_values = coming_a_values

        self.plotting_widgets_list = [self.ui.system_phase_response_widget, self.ui.system_z_plane_widget, self.ui.library_phase_response_widget, self.ui.library_z_plane_widget]

        # print("zeros and poles in dialog ", complex_zeroes_original, complex_poles_original)

        self.init_graphs_lists_transferfunctions()

        #init graphs,z plane, cascaded system / a list (system) z plane, first, old a values
        ############################################################################################################################
        self.ui.system_z_plane_widget.setXRange(-2.2, 2.2)
        self.ui.system_z_plane_widget.setYRange(-1.22, 1.22)
        self.ui.library_z_plane_widget.setXRange(-2.2, 2.2)
        self.ui.library_z_plane_widget.setYRange(-1.22, 1.22)

        if (self.coming_a_values):
            self.build_system_from_coming_a_values(self.coming_a_values)
            poles_all_pass = [complex(p) for p in self.coming_a_values]
            zeros_all_pass = self.zeros_all_pass = [1/p.conjugate() for p in self.poles_all_pass]
            # print(" self.poles_all_pass ", self.poles_all_pass, " self.zeros_all_pass ", self.zeros_all_pass)
            self.poles_scatter_all_pass = pg.ScatterPlotItem([pole.real for pole in poles_all_pass], [pole.imag for pole in poles_all_pass], symbol = 'x', size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
            self.zeros_scatter_all_pass = pg.ScatterPlotItem([zero.real for zero in zeros_all_pass], [zero.imag for zero in zeros_all_pass], symbol = 'o', size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
            self.ui.system_z_plane_widget.addItem(self.poles_scatter_all_pass)
            self.ui.system_z_plane_widget.addItem(self.zeros_scatter_all_pass)
            
        # print("MyDialog.coming a values ", self.coming_a_values)

        self.ui.add_a_value_button.clicked.connect(self.add_a_to_list_and_system)

        self.ui.library_list.itemClicked.connect(self.when_select_a)
        self.ui.library_list.currentItemChanged.connect(self.when_select_a)
        self.ui.system_list.itemClicked.connect(self.when_select_a)
        self.ui.system_list.currentItemChanged.connect(self.when_select_a)

        self.ui.library_list.itemDoubleClicked.connect(self.doubleclick_a_from_list)
        self.ui.system_list.itemDoubleClicked.connect(self.remove_a_from_list)
        
        self.ui.buttonBox.accepted.connect(self.return_to_main_window)
        self.ui.create_new_all_pass_btn.clicked.connect(self.init_graphs_lists_transferfunctions)

    def init_graphs_lists_transferfunctions(self):
        # init graphs, clear graphs
        ############################################################################################################################
        for widget in self.plotting_widgets_list:
            widget.plotItem.clear()

        # init graphs, phase respone, plot the main window phase response 
        ############################################################################################################################
        omegas, freq_response = signal.freqz_zpk(self.complex_zeroes_original, self.complex_poles_original, 1)
        h_phase = np.unwrap(np.angle(freq_response))
        omegas = np.around(omegas, 3)
        self.applied_filter_phase_response_plot_data_item = pg.PlotDataItem(omegas, h_phase)
        self.ui.system_phase_response_widget.addItem(self.applied_filter_phase_response_plot_data_item)
        self.system_phase_response_PltDataItem = None

        # init graphs, z planes constant elements (grid, axis, unit circle)
        ############################################################################################################################
        x_axis_line_1 = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w'))
        x_axis_line_2 = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w'))

        y_axis_line_1 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w'))
        y_axis_line_2 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w'))

        theta = np.linspace(0, 2*np.pi, 360)
        x = np.cos(theta)
        y = np.sin(theta)
        unit_circle_1 = pg.PlotDataItem(x, y)
        unit_circle_2 = pg.PlotDataItem(x, y)

        grid_1 = pg.GridItem()
        grid_1.setTextPen ( None)

        grid_2 = pg.GridItem()
        grid_2.setTextPen(None)

        z_planes_constant_elements_1 = [x_axis_line_1, y_axis_line_1,unit_circle_1 , grid_1]
        z_planes_constant_elements_2 = [x_axis_line_2, y_axis_line_2,unit_circle_2 , grid_2]
        # print("  self.ui.library_z_plane_widget ", self.ui.library_z_plane_widget)
        # print ( " system_z_plane_widget ", self.ui.system_z_plane_widget, "  self.system_z_plane_widget.plotitem ", self.ui.system_z_plane_widget.plotItem)
        
        for const_item_1, const_item_2 in zip(z_planes_constant_elements_1, z_planes_constant_elements_2):
            try:
                self.ui.library_z_plane_widget.addItem(const_item_1)
            except Exception as e:
                print(f"Error adding item: {e}")
            self.ui.system_z_plane_widget.addItem(const_item_2)

        # init graphs, z plane, plot the main window z plane, constant
        ############################################################################################################################
        poles_real = [pole.real for pole in self.complex_poles_original]
        poles_imag = [pole.imag for pole in self.complex_poles_original]
        zeroes_real = [zero.real for zero in self.complex_zeroes_original]
        zeroes_imag = [zero.imag for zero in self.complex_zeroes_original]

        # Plot the poles and zeroes as scatter plots
        poles_scatter = pg.ScatterPlotItem(poles_real, poles_imag,symbol = 'x', size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 255))
        zeroes_scatter = pg.ScatterPlotItem(zeroes_real, zeroes_imag, symbol = 'o',size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 255))
        # Create a legend
        #legend = pg.LegendItem(size = (5,5), labelTextSize='5pt')

        # Add items to the legend
        # legend.addItem(poles_scatter, 'Poles digital filter')
        # legend.addItem(zeroes_scatter, 'Zeros digital filter')

        # Add the legend to the plot
        # self.ui.system_z_plane_widget.addItem(legend)

        # Add the scatter plots to the plot
        self.ui.system_z_plane_widget.addItem(poles_scatter)
        self.ui.system_z_plane_widget.addItem(zeroes_scatter)
        self.ui.system_z_plane_widget.setXRange(-2.2,2.2)
        self.ui.system_z_plane_widget.setYRange(-1.33,1.33)

        #init graphs, phase response, plotting the phase response of a chosen a value, changes by choosing a single a value
        ############################################################################################################################
        self.all_pass_filter_for_a_PltDataItm = None
        
        #init graphs,z plane, plot the chosen a (library) z plane
        ############################################################################################################################
        self.chosen_a_pole, self.chosen_a_zero = complex(0), complex(0)
        self.chosen_a_z_plane_poles , self.chosen_a_z_plane_zeros = None, None

        # init graphs, phase response, set labels of phase response graphs
        ############################################################################################################################
        for i in range(0,3,2):
            self.plotting_widgets_list[i].setLabel('bottom', "w normalized")
            self.plotting_widgets_list[i].setLabel('left', "Phase Response")
            # self.plotting_widgets_list[i].setXRange(-3, 3)
            # self.plotting_widgets_list[i].setYRange(-3,3)

        # init system list and transfer
        ############################################################################################################################
        self.ui.system_list.clear()
        
        # init transfer function and ((system list)), just update the transfer function of the cascaded system to be the one of the filter in the main window
        ############################################################################################################################
        original_filter = signal.TransferFunction(np.poly(self.complex_zeroes_original),
                                                    np.poly(self.complex_poles_original))
        
        self.transfer_function_cascaded_system = original_filter
        self.original_transfer_function = original_filter

        # init all pass 
        ############################################################################################################################
        self.poles_scatter_all_pass, self.zeros_scatter_all_pass = None, None
        self.poles_all_pass, self.zeros_all_pass  = [], []

    def build_system_from_coming_a_values(self, coming_a_values_to_build_system):
        for a_value in coming_a_values_to_build_system:
            self.add_a_to_list_and_system(a_value)
        
    def return_to_main_window(self):
        #TODO: return zeros and poles
        poles_to_main = self.poles_all_pass
        system_transfer_function_to_main_window = self.transfer_function_cascaded_system
        # print( " poles_to_main ", poles_to_main, " zeros to main hopefuly ", self.zeros_all_pass )
        # print ( " system_transfer_function_to_main_window ",system_transfer_function_to_main_window )

        return poles_to_main, system_transfer_function_to_main_window

    def add_a_to_list_and_system(self, lib = False):
        if (lib == False) :
            value = (self.ui.add_a_value_lnedit.text()) 
        else: value = lib

        # print(" value ", value)


        if (value != '0' and value != '1'):
            # add to the list, clear line edit
            self.ui.system_list.addItem(str(complex(value)))
            # print all items in the list
            for i in range(self.ui.system_list.count()):
                # print("list values after adding " , self.ui.system_list.item(i).text())
                pass
            self.ui.add_a_value_lnedit.clear()

            # Assuming complex_zeroes_original and complex_poles_original are initialized somewhere in your code
            allpass_filter = self.create_allpass_filter(value)

            # Cascade the all-pass filter with the original filter
            cascaded_num, cascaded_den = signal.convolve(self.transfer_function_cascaded_system.num, allpass_filter.num), signal.convolve(
                self.transfer_function_cascaded_system.den, allpass_filter.den)

            cascaded_system = signal.TransferFunction(cascaded_num, cascaded_den)
            self.transfer_function_cascaded_system = cascaded_system
            #print("cascaded system ",cascaded_system )

            # Frequency response of the original and cascaded systems
            frequencies, response_cascaded = signal.freqz(cascaded_system.num, cascaded_system.den)

            # Plot phase response
            if (not self.system_phase_response_PltDataItem):
                self.system_phase_response_PltDataItem = pg.PlotDataItem(frequencies, np.unwrap(np.angle(response_cascaded)), pen='r', name='Cascaded System')
                self.ui.system_phase_response_widget.addItem( self.system_phase_response_PltDataItem)
            else:
                self.system_phase_response_PltDataItem.setData(frequencies,  np.unwrap(np.angle(response_cascaded)))

            # plot z plane
            self.poles_all_pass.append(complex(value)); self.zeros_all_pass.append(1/(complex(value).conjugate()))
            if (not self.poles_scatter_all_pass and not self.zeros_scatter_all_pass):
                self.poles_scatter_all_pass = pg.ScatterPlotItem([pole.real for pole in self.poles_all_pass], [pole.imag for pole in self.poles_all_pass], symbol = 'x', size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
                self.zeros_scatter_all_pass = pg.ScatterPlotItem([zero.real for zero in self.zeros_all_pass], [zero.imag for zero in self.zeros_all_pass], symbol = 'o', size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
                self.ui.system_z_plane_widget.addItem(self.poles_scatter_all_pass)
                self.ui.system_z_plane_widget.addItem(self.zeros_scatter_all_pass)
            else:
                self.poles_scatter_all_pass.setData([pole.real for pole in self.poles_all_pass], [pole.imag for pole in self.poles_all_pass])
                self.zeros_scatter_all_pass.setData([zero.real for zero in self.zeros_all_pass], [zero.imag for zero in self.zeros_all_pass])
                    


    def remove_a_from_list(self, a_value_to_be_removed_list_object):
        a_value_to_be_removed = a_value_to_be_removed_list_object.text()
        # print("a value to be removed ", a_value_to_be_removed)
        # update the zeros and poles lists
        self.poles_all_pass.remove(complex(a_value_to_be_removed)), self.zeros_all_pass.remove(1/complex(a_value_to_be_removed).conjugate())
        freqs_after_removal, system_after_removal_response = signal.freqz_zpk(self.zeros_all_pass + self.complex_zeroes_original, self.poles_all_pass + self.complex_poles_original, 1) 

        # print(" poles and zeros after removal ", self.poles_all_pass, self.zeros_all_pass)
        self.poles_scatter_all_pass.setData([pole.real for pole in self.poles_all_pass], [pole.imag for pole in self.poles_all_pass])
        self.zeros_scatter_all_pass.setData([zero.real for zero in self.zeros_all_pass], [zero.imag for zero in self.zeros_all_pass])
            
        # Plot phase response and z plane
        self.system_phase_response_PltDataItem.setData(freqs_after_removal, np.unwrap(np.angle(system_after_removal_response)))

        self.ui.system_list.takeItem(self.ui.system_list.row(a_value_to_be_removed_list_object))

    def doubleclick_a_from_list(self, item):
        lib = item.text()
        self.add_a_to_list_and_system(lib)
    # Function to create TransferFunction for the given Z-transform
    def create_allpass_filter(self,a):
        # Convert string to complex number if necessary
        a = complex(a)
        num = [- a.conjugate(), 1]  # Use complex conjugate of 'a' in the numerator
        den = [1,- a]
        return signal.TransferFunction(num, den)

    def when_select_a(self, a_value_allpass : QListWidgetItem ):

        if a_value_allpass:
            # print("a_value_allpass in when_select_a ", a_value_allpass.text())
            a_value_allpass = a_value_allpass.text()
            allpass_filter = self.create_allpass_filter(a_value_allpass)
            omegas, allpass_response = signal.freqz(allpass_filter.num, allpass_filter.den)
            phase = np.unwrap(np.angle(allpass_response))
            mag = np.abs(allpass_response)
            if (not self.all_pass_filter_for_a_PltDataItm): self.all_pass_filter_for_a_PltDataItm=pg.PlotDataItem(omegas, phase); self.ui.library_phase_response_widget.addItem(self.all_pass_filter_for_a_PltDataItm)
            else : self.all_pass_filter_for_a_PltDataItm.setData(omegas, phase)

            if(not  self.chosen_a_z_plane_poles and not self.chosen_a_z_plane_zeros ): 
                self.chosen_a_z_plane_poles = pg.ScatterPlotItem([complex(a_value_allpass).real], [complex(a_value_allpass).imag],symbol = 'x', size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255) )
                self.chosen_a_z_plane_zeros = pg.ScatterPlotItem([(1/complex(a_value_allpass).conjugate()).real], [(1/complex(a_value_allpass).conjugate()).imag],symbol = 'o', size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255) )
                self.ui.library_z_plane_widget.addItem(self.chosen_a_z_plane_poles)
                self.ui.library_z_plane_widget.addItem(self.chosen_a_z_plane_zeros)
            else:
                self.chosen_a_z_plane_poles.setData([complex(a_value_allpass).real], [complex(a_value_allpass).imag])
                self.chosen_a_z_plane_zeros.setData([(1/complex(a_value_allpass).conjugate()).real], [(1/complex(a_value_allpass).conjugate()).imag])

class CustomPaddingWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setMouseTracking(True)

        self.amplitude = 1e-10

    def leaveEvent(self, a0: QEvent | None) -> None:
        main_window_instance.padding_timer.stop()
        return super().leaveEvent(a0)

    def enterEvent(self, event: QEnterEvent | None) -> None:

        main_window_instance.ui.signal_input_graph.plotItem.clear()
        main_window_instance.ui.filter_output_graph.plotItem.clear()
        
        main_window_instance.ui.signal_input_graph.setXRange(min=-50, max=50)
        main_window_instance.ui.filter_output_graph.setXRange(min=-50, max=50)
        # main_window_instance.ui.filter_output_graph.setYRange(min = -110, max = 50)
        # main_window_instance.ui.signal_input_graph.setYRange(min = -110, max = 50)
    
        main_window_instance.data = []
        main_window_instance.filtered_pad_sig = np.zeros(10000, dtype=complex)
        main_window_instance.time = np.arange(0, 10001, 1)
        main_window_instance.count = 0
        main_window_instance.padding_timer.start()

        return super().enterEvent(event)
        
    def mouseMoveEvent(self, event):
        self.amplitude = (310 - event.position().y() )/50

class CustomGraphItem(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)

    def setData(self, **kwds):
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)

        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        
    def mouseDragEvent(self, ev):
        # drag using the left button
        if ev.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            # no points to drag
            if len(pts) == 0:
                ev.ignore()
                return
            
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        
        # remove the closest point (the positions are not exact) and add the new position
        elif ev.isFinish():
            closest_pt = MyMainWindow.find_closest_point(ev.buttonDownPos(), main_window_instance.zeroes_coords + main_window_instance.poles_coords)
            list = main_window_instance.coords[not (any(np.array_equal(closest_pt, coord) for coord in main_window_instance.zeroes_coords))]

            list.remove(closest_pt.tolist())
            list.append([ev.pos().x(), ev.pos().y()])
            main_window_instance.plot_responses()
            self.dragPoint = None
            return
        
        else: # we don't do anything while the event is happening only at the start/end
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()
    
    def clicked(self, pts, spotItem):
        list = main_window_instance.coords[
            [spotItem[0].viewPos().x(), spotItem[0].viewPos().y()] in main_window_instance.poles_coords
            ]
        # print(f"list to remove from {list}")
        # print(f"item to be removed {[spotItem[0].viewPos().x(), spotItem[0].viewPos().y()]}")
        list.remove([spotItem[0].viewPos().x(), spotItem[0].viewPos().y()])

        main_window_instance.addPtsToGraph()
        main_window_instance.setup_filter_from_zeroes_poles()
        main_window_instance.plot_responses()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window_instance = MyMainWindow()
    sys.exit(app.exec())
