# Form implementation generated from reading ui file 'all_pass.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(780, 1042)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.add_a_value_label = QtWidgets.QLabel(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_a_value_label.sizePolicy().hasHeightForWidth())
        self.add_a_value_label.setSizePolicy(sizePolicy)
        self.add_a_value_label.setObjectName("add_a_value_label")
        self.horizontalLayout.addWidget(self.add_a_value_label)
        self.add_a_value_lnedit = QtWidgets.QLineEdit(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_a_value_lnedit.sizePolicy().hasHeightForWidth())
        self.add_a_value_lnedit.setSizePolicy(sizePolicy)
        self.add_a_value_lnedit.setObjectName("add_a_value_lnedit")
        self.horizontalLayout.addWidget(self.add_a_value_lnedit)
        self.add_a_value_button = QtWidgets.QPushButton(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_a_value_button.sizePolicy().hasHeightForWidth())
        self.add_a_value_button.setSizePolicy(sizePolicy)
        self.add_a_value_button.setObjectName("add_a_value_button")
        self.horizontalLayout.addWidget(self.add_a_value_button)
        self.create_new_all_pass_btn = QtWidgets.QPushButton(parent=Dialog)
        self.create_new_all_pass_btn.setObjectName("create_new_all_pass_btn")
        self.horizontalLayout.addWidget(self.create_new_all_pass_btn)
        self.horizontalLayout_7.addLayout(self.horizontalLayout)
        self.gridLayout_2.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.system_lbl = QtWidgets.QLabel(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.system_lbl.sizePolicy().hasHeightForWidth())
        self.system_lbl.setSizePolicy(sizePolicy)
        self.system_lbl.setObjectName("system_lbl")
        self.horizontalLayout_6.addWidget(self.system_lbl)
        self.library_list_lbl = QtWidgets.QLabel(parent=Dialog)
        self.library_list_lbl.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.library_list_lbl.sizePolicy().hasHeightForWidth())
        self.library_list_lbl.setSizePolicy(sizePolicy)
        self.library_list_lbl.setObjectName("library_list_lbl")
        self.horizontalLayout_6.addWidget(self.library_list_lbl)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.system_list = QtWidgets.QListWidget(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.system_list.sizePolicy().hasHeightForWidth())
        self.system_list.setSizePolicy(sizePolicy)
        self.system_list.setObjectName("system_list")
        self.horizontalLayout_5.addWidget(self.system_list)
        self.library_list = QtWidgets.QListWidget(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.library_list.sizePolicy().hasHeightForWidth())
        self.library_list.setSizePolicy(sizePolicy)
        self.library_list.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked|QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed|QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
        self.library_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.library_list.setObjectName("library_list")
        item = QtWidgets.QListWidgetItem()
        self.library_list.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.library_list.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.library_list.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.library_list.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.library_list.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.library_list.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.library_list.addItem(item)
        self.horizontalLayout_5.addWidget(self.library_list)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.system_phase_response_lbl = QtWidgets.QLabel(parent=Dialog)
        self.system_phase_response_lbl.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.system_phase_response_lbl.sizePolicy().hasHeightForWidth())
        self.system_phase_response_lbl.setSizePolicy(sizePolicy)
        self.system_phase_response_lbl.setObjectName("system_phase_response_lbl")
        self.horizontalLayout_4.addWidget(self.system_phase_response_lbl)
        self.library_phase_response_lbl = QtWidgets.QLabel(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.library_phase_response_lbl.sizePolicy().hasHeightForWidth())
        self.library_phase_response_lbl.setSizePolicy(sizePolicy)
        self.library_phase_response_lbl.setObjectName("library_phase_response_lbl")
        self.horizontalLayout_4.addWidget(self.library_phase_response_lbl)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.system_phase_response_widget = PlotWidget(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.system_phase_response_widget.sizePolicy().hasHeightForWidth())
        self.system_phase_response_widget.setSizePolicy(sizePolicy)
        self.system_phase_response_widget.setObjectName("system_phase_response_widget")
        self.horizontalLayout_3.addWidget(self.system_phase_response_widget)
        self.library_phase_response_widget = PlotWidget(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.library_phase_response_widget.sizePolicy().hasHeightForWidth())
        self.library_phase_response_widget.setSizePolicy(sizePolicy)
        self.library_phase_response_widget.setObjectName("library_phase_response_widget")
        self.horizontalLayout_3.addWidget(self.library_phase_response_widget)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.system_z_plane_lbl = QtWidgets.QLabel(parent=Dialog)
        self.system_z_plane_lbl.setObjectName("system_z_plane_lbl")
        self.horizontalLayout_2.addWidget(self.system_z_plane_lbl)
        self.library_z_plane_lbl = QtWidgets.QLabel(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.library_z_plane_lbl.sizePolicy().hasHeightForWidth())
        self.library_z_plane_lbl.setSizePolicy(sizePolicy)
        self.library_z_plane_lbl.setObjectName("library_z_plane_lbl")
        self.horizontalLayout_2.addWidget(self.library_z_plane_lbl)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.system_z_plane_widget = PlotWidget(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.system_z_plane_widget.sizePolicy().hasHeightForWidth())
        self.system_z_plane_widget.setSizePolicy(sizePolicy)
        self.system_z_plane_widget.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)
        self.system_z_plane_widget.setObjectName("system_z_plane_widget")
        self.gridLayout.addWidget(self.system_z_plane_widget, 0, 0, 1, 1)
        self.library_z_plane_widget = PlotWidget(parent=Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.library_z_plane_widget.sizePolicy().hasHeightForWidth())
        self.library_z_plane_widget.setSizePolicy(sizePolicy)
        self.library_z_plane_widget.setObjectName("library_z_plane_widget")
        self.gridLayout.addWidget(self.library_z_plane_widget, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_5.addLayout(self.verticalLayout_3)
        self.verticalLayout_6.addLayout(self.verticalLayout_5)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_6.addWidget(self.buttonBox)
        self.gridLayout_2.addLayout(self.verticalLayout_6, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.add_a_value_label.setText(_translate("Dialog", "A value"))
        self.add_a_value_button.setText(_translate("Dialog", "Add"))
        self.create_new_all_pass_btn.setText(_translate("Dialog", "Create New All Pass Filter"))
        self.system_lbl.setText(_translate("Dialog", "Constructed All-Pass Filter"))
        self.library_list_lbl.setText(_translate("Dialog", "All-Pass a values library."))
        __sortingEnabled = self.library_list.isSortingEnabled()
        self.library_list.setSortingEnabled(False)
        item = self.library_list.item(0)
        item.setText(_translate("Dialog", "(0+2j)"))
        item = self.library_list.item(1)
        item.setText(_translate("Dialog", "(1+2j)"))
        item = self.library_list.item(2)
        item.setText(_translate("Dialog", "(0.2+0.8j)"))
        item = self.library_list.item(3)
        item.setText(_translate("Dialog", "(-3+0.1j)"))
        item = self.library_list.item(4)
        item.setText(_translate("Dialog", "(5+0j)"))
        item = self.library_list.item(5)
        item.setText(_translate("Dialog", "(1.5+0j)"))
        item = self.library_list.item(6)
        item.setText(_translate("Dialog", "(-1.5+0j)"))
        self.library_list.setSortingEnabled(__sortingEnabled)
        self.system_phase_response_lbl.setText(_translate("Dialog", "System Phase Response Plot."))
        self.library_phase_response_lbl.setText(_translate("Dialog", "Chosen a value - Phase Respone Plot."))
        self.system_z_plane_lbl.setText(_translate("Dialog", "System Z-Plane Plot."))
        self.library_z_plane_lbl.setText(_translate("Dialog", "Chosen a value - Z-Plane Plot."))
from pyqtgraph import PlotWidget
