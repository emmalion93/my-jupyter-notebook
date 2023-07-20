import os
import hublib.use

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    from IPython.display import clear_output, display, HTML
    import io
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import string
    import pandas as pd
    import seaborn as sns
    
    from srim import SR, Ion, Layer, Target
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

user_data_folder = "../../../srqueriertool_data"

def data_check():
    #check for user data folder
    if not os.path.exists(user_data_folder):
        os.makedirs(user_data_folder)

data = pd.read_csv("data/stackup_data.csv", index_col = [0,1])
user_data = pd.DataFrame()

if os.path.exists(f"{user_data_folder}/user_data.csv"):
    user_data = pd.read_csv(f"{user_data_folder}/user_data.csv", index_col = [0,1])
    if len(user_data) == 0:
        os.remove(f"{user_data_folder}/user_data.csv") #invalid results sometimes occurs
    data = data.append(user_data)
    data=data.drop_duplicates()

ptoe = pd.read_csv("data/ptoe.csv", index_col=0)
ion_lookup = ptoe.Symbol.to_dict()

#t - energy, range, lat, long, let
def lookup(t, mat, ion, target):
    if t == "dE/dx":
        data_slice = data.loc[(mat, t)][ion]
        target = data_slice.max() - target #important to calculate this before shifting
        data_slice = data_slice.max() - data_slice
    else:
        data_slice = data.loc[(mat, ion)][t]
        
    index_high = np.searchsorted(data_slice.values, target)
    index_low = index_high-1
    
    #index range checks (before interpolation calulation
    if index_high == len(data_slice):
        index_high = len(data_slice)-1
        index_low = index_high-1
    if index_high == 0:
        index_high = 1
        index_low = 0
        
    interp = np.interp(target, (data_slice[index_low], data_slice[index_high]), (0, 1))
    return lookup_interp(mat, ion, index_low, index_high, interp)

def lookup_interp(mat, ion, index_low, index_high, interp):
    mat_slice_low = data.loc[(mat, ion)].iloc[index_low]
    mat_slice_high = data.loc[(mat, ion)].iloc[index_high]
    return {
        'energy': np.interp(interp, (0, 1), (mat_slice_low["Energy"], mat_slice_high["Energy"])), #energy doesn't use material energy lookup but global energy lookup
        'range': np.interp(interp, (0, 1), (mat_slice_low["Range"], mat_slice_high["Range"])),
        'let': np.interp(interp, (0, 1), (mat_slice_low["dE/dx"], mat_slice_high["dE/dx"])),
        'lat': np.interp(interp, (0, 1), (mat_slice_low["Lateral Straggling"], mat_slice_high["Lateral Straggling"])),
        'long': np.interp(interp, (0, 1), (mat_slice_low["Long. Straggling"], mat_slice_high["Long. Straggling"])),
        'index_low': index_low,
        'index_high': index_high,
        'interp': interp
    }

#returns exit_energy, remaining_range (in material), entrance_let, exit_let, valid, #DEBUG ONLY# entrance_info, exit_info
def calc_layer(mat, ion, target_energy, layer_thickness, debug=False):
    try:
        entrance_info = lookup("Energy", mat, ion, target_energy)
        exit_range = entrance_info["range"] - layer_thickness
        exit_info = lookup("Range", mat, ion, exit_range)
        ret = (exit_info["energy"], exit_range, entrance_info["let"], exit_info["let"], exit_range > 0)

        if debug:
            ret = ret + (entrance_info, exit_info)

        return ret
    except:
        return (0, 0, 0, 0, False) #If an error occured the values returned don't matter but we want to set valid to False

class FixedSizedLayout(widgets.Layout):
    def __init__(self, min_width='190px', max_width='190px'):
        super(FixedSizedLayout, self).__init__(min_width=min_width, max_width=max_width)
        
class FixedIconButtonLayout(widgets.Layout):
    def __init__(self, min_width='37.5px', max_width='37.5px', border='1px solid #000000'):
        super(FixedIconButtonLayout, self).__init__(min_width=min_width, max_width=max_width, border=border)

class ColLayout(widgets.Layout):
    def __init__(self, min_width='150px', max_width='150px', margin="0px 5px 0px 0px"):
        super(ColLayout, self).__init__(min_width=min_width, max_width=max_width, margin=margin)

class ItemLayout(widgets.Layout):
    def __init__(self, min_width='100%', max_width='100%', margin="0px 0px 5px 0px"):
        super(ItemLayout, self).__init__(min_width=min_width, max_width=max_width, margin=margin)

class ISEEUWidget(widgets.Tab):
    def __init__(self):
        self.names = ["Multi-Layer Query", "Single-Layer Query", "Data Viewer", "Import Data", "PySRIM"]

        self.multiquery = MultiQuery(self)
        self.singlequery = SingleQuery(self)
        self.datatab = DataWidget(self)
        self.importtab = ImportTab(self)
        self.pysrim = PySRIM(self)

        self.children = [self.multiquery, self.singlequery, self.datatab, self.importtab, self.pysrim]

        for n, name in enumerate(self.names):
            self.set_title(n, name)

        super(ISEEUWidget, self).__init__(self.children)
        
    def display(self):
        return self
    
    def update(self):
        self.multiquery.update()
        self.singlequery.update()
        self.datatab.update()

class MultiQuery(widgets.VBox):
    def __init__(self, parent):
        display(HTML('''
            <style>
                .layer_input input {
                    background-color: #F1EB9D !important;
                }
                .top_row {
                    font-family: Verdana, sans-serif;
                    font-size: 95%;
                    font-weight: bold;
                }
                .top_row_dd1 select {
                    background-color: #AAE1FC !important;
                }
                .top_row_input input {
                    background-color: #FFE1E1 !important;
                }
                .calcd_value_generic_nocolor input {
                    font-family: DejaVu Sans Mono, monospace;
                    font-weight: 950;
                }
                .mat_dd_layer01 select { background-color: #99FFFA !important; }
                .mat_dd_layer02 select { background-color: #99FFC8 !important; }
                .mat_dd_layer03 select { background-color: #99FF96 !important; }
                .mat_dd_layer04 select { background-color: #99FF64 !important; }
                .mat_dd_layer05 select { background-color: #99FF32 !important; }
            </style>
        '''))
        
        style = {'description_width': 'initial'}
        
        self.ion_dropdown = widgets.Dropdown(
            options=data.index.unique(level="ion"), 
            value="H", 
            description='Ion:',
            disabled=False, 
            style = style
        )
        self.ion_energy_selector = widgets.BoundedFloatText(
            description='Ion Energy:',
            disabled=False,
            value=300,
            min=0,
            max=data["Energy"].max(),
            step = 0.1,
            style = style
        )
        
        self.ion_dropdown.add_class("top_row_dd1")
        self.ion_energy_selector.add_class("top_row_input")
        
        self.add_button = widgets.Button(icon="plus", disabled=False, button_style='success', layout=FixedIconButtonLayout())
        self.add_button.on_click(self.add_button_clicked)
        
        self.layer_container = LayerContainer(self.ion_dropdown, self.ion_energy_selector)
        self.parent = parent

        ion = widgets.HBox([self.ion_dropdown, self.ion_energy_selector]).add_class("top_row")
        headers = widgets.HBox([
            widgets.Label(value="Material",                       layout=FixedSizedLayout()),
            widgets.Label(value="Thickness $\;mm$",               layout=FixedSizedLayout()),
            widgets.Label(value="Exit Energy $\;MeV/u$",          layout=FixedSizedLayout()),
            widgets.Label(value="Residual Range $\;mm$",          layout=FixedSizedLayout()),
            widgets.Label(value="Entrance LET $\;MeV/(mg/cm^2)$", layout=FixedSizedLayout()),
            widgets.Label(value="Exit LET $\;MeV/(mg/cm^2)$",     layout=FixedSizedLayout()),
            self.add_button
        ])
        super(MultiQuery, self).__init__([ion, headers, self.layer_container])
    
    def update(self):
        self.ion_dropdown.options = data.index.unique(level="ion")
        self.layer_container.update()
    
    def add_button_clicked(self, b):
        self.layer_container.add_layer_at(0)

class LayerContainer(widgets.VBox):
    def __init__(self, ion_dropdown=None, ion_energy_selector=None):
        style = {
            'description_width': 'initial', 
            'border-color': 'black'
        }
        
        self.ion_dropdown = ion_dropdown
        self.ion_energy_selector = ion_energy_selector
        
        if ion_dropdown and ion_energy_selector:
            ion_dropdown.observe(self.on_value_change, names='value')
            ion_energy_selector.observe(self.on_value_change, names='value')
        
        self.layers = []
        self.add_layer_at(0)
        
        super(LayerContainer, self).__init__(children=self.layers)
    
    def add_layer_at(self, index):
        layer = MultiQueryLayer(self, index)
        layer.material.observe(self.on_value_change, names='value')
        layer.thickness.observe(self.on_value_change, names='value')
        # self.layers.append(layer)
        self.layers.insert(index, layer)
        self.update_children()
    
    def remove_layer(self, index):
        self.layers.pop(index)
        self.update_children()
    
    def move_layer_up(self, index):
        self.layers.insert(index-1, self.layers.pop(index))
        self.update_children()
    
    def move_layer_down(self, index):
        self.layers.insert(index+1, self.layers.pop(index))
        self.update_children()
    
    def update_children(self):
        self.children = tuple(self.layers)
        for layer_n, layer in enumerate(self.layers):
            layer.update_controls(layer_n)
        self.calculate()
    
    def on_value_change(self, change):
        self.calculate()
        
    def get_layer_count(self):
        return len(self.layers)
    
    def calculate(self):
        if self.ion_energy_selector:
            energy = self.ion_energy_selector.value
            for layer in self.layers:
                energy, residual_range, entrance_let, exit_let, valid = layer.calc_layer(self.ion_dropdown.value, energy)
                
                if not valid:
                    energy = 0

    def update(self):
        for layer in self.layers:
            layer.update()
        
class MultiQueryLayer(widgets.HBox):
    def __init__(self, parent_container: LayerContainer, index: int):
        # get layer color based on layer count
        mat_dd_layer_colors = [
            "mat_dd_layer01",
            "mat_dd_layer02",
            "mat_dd_layer03",
            "mat_dd_layer04",
            "mat_dd_layer05"
        ]
        mat_dd_layer_colors += mat_dd_layer_colors[::-1]
        
        self.parent_container = parent_container
        
        self.material = widgets.Dropdown(
            value="Silicon", 
            options=data.index.unique(level="mat"), 
            disabled=False, 
            layout = FixedSizedLayout()
        )
        
        self.thickness = widgets.BoundedFloatText(
            value = 10,
            min = 0,
            max = data["Range"].max(),
            step = 0.1,
            redout = False,
            disabled = False,
            layout = FixedSizedLayout()
        )
        
        self.add_button = widgets.Button(icon="plus", disabled=False, button_style='success', layout=FixedIconButtonLayout())
        self.move_up_button = widgets.Button(icon="arrow-up", disabled=False, button_style='info', layout=FixedIconButtonLayout())
        self.move_down_button = widgets.Button(icon="arrow-down", disabled=False, button_style='info', layout=FixedIconButtonLayout())
        self.delete_button = widgets.Button(icon="trash", disabled=False, button_style='danger', layout=FixedIconButtonLayout())
        
        self.add_button.on_click(self._add_button_clicked)
        self.move_up_button.on_click(self._move_up_button_clicked)
        self.move_down_button.on_click(self._move_down_button_clicked)
        self.delete_button.on_click(self._delete_button_clicked)
        
        self.exit_energy_text = widgets.Text(value="", disabled=True,layout = FixedSizedLayout())
        self.residual_range_text = widgets.Text(value="", disabled=True, layout = FixedSizedLayout())
        self.entrance_let_text = widgets.Text(value="", disabled=True, layout = FixedSizedLayout())
        self.exit_let_text = widgets.Text(value="", disabled=True, layout = FixedSizedLayout())
        
        self.exit_energy_text.style.text_color = self.residual_range_text.style.text_color = self.entrance_let_text.style.text_color = self.exit_let_text.style.text_color = 'black'
        
        c = [self.material.add_class(mat_dd_layer_colors[self.parent_container.get_layer_count()-1 if self.parent_container.get_layer_count() < 10 else self.parent_container.get_layer_count() % 10]), 
             self.thickness.add_class("layer_input"), 
             self.exit_energy_text.add_class("calcd_value_generic_nocolor"), 
             self.residual_range_text.add_class("calcd_value_generic_nocolor"), 
             self.entrance_let_text.add_class("calcd_value_generic_nocolor"), 
             self.exit_let_text.add_class("calcd_value_generic_nocolor"),
             self.add_button, 
             self.move_up_button, 
             self.move_down_button, 
             self.delete_button]
        
        if self.parent_container.get_layer_count() <= 1:
            self.delete_button.disabled = True
        
        self.update_controls(index)
            
        super(MultiQueryLayer, self).__init__(children=c)
    
    def _add_button_clicked(self, b):
        self.parent_container.add_layer_at(self.index+1)
    
    def _move_up_button_clicked(self, b):
        self.parent_container.move_layer_up(self.index)
        
    def _move_down_button_clicked(self, b):
        self.parent_container.move_layer_down(self.index)
    
    def _delete_button_clicked(self, b):
        self.parent_container.remove_layer(self.index)
    
    def update_controls(self, index):
        self.index = index
        
        self.move_down_button.disabled = self.delete_button.disabled = self.move_up_button.disabled = False
        
        if index == 0:
            self.move_up_button.disabled = True
        if index >= self.parent_container.get_layer_count()-1:
            self.move_down_button.disabled = True
        if self.parent_container.get_layer_count() <= 1:
            self.delete_button.disabled = True
    
    def calc_layer(self, ion, energy, update=True):
        exit_energy, residual_range, entrance_let, exit_let, valid = calc_layer(self.material.value, ion, energy, self.thickness.value)
        
        if update:
            self.exit_energy_text.value = str(exit_energy)
            self.residual_range_text.value = str(residual_range)
            self.entrance_let_text.value = str(entrance_let)
            self.exit_let_text.value = str(exit_let)

            #388e3c -> #CDE3CE & #d3302f -> #F4CBCB (http://origin.filosophy.org/code/online-tool-to-lighten-color-without-alpha-channel/)
            bg_color = "#CDE3CE" if valid else "#F4CBCB"

            self.exit_energy_text.style.background = self.residual_range_text.style.background = self.entrance_let_text.style.background = self.exit_let_text.style.background = bg_color

            if not valid:
                self.exit_energy_text.value = self.residual_range_text.value = self.entrance_let_text.value = self.exit_let_text.value = "NaN"
        
        return exit_energy, residual_range, entrance_let, exit_let, valid
    
    def update(self):
        self.material.options = data.index.unique(level="mat")

class AttachedFile(widgets.HBox):
    def __init__(self, file, parent):
        self.file = file
        self.file_name = widgets.Text(value=file["metadata"]["name"], disabled=True)
        self.file_remove = widgets.Button(icon="times", button_style='danger')
        self.file_remove.on_click(self._remove_file)
        self.parent = parent
        
        super(AttachedFile, self).__init__([self.file_name, self.file_remove])
    
    def _remove_file(self, b):
        self.parent.remove_file(self)

class ImportTab(widgets.VBox):
    def __init__(self, parent):
        self.data = data
        self.attach_files = widgets.FileUpload(description='Attach', icon="", accept='', multiple=True)
        self.attach_files.observe(self._attach_file_button_changed, names='value')
        self.attached_files = []
        self.parent = parent
        
        self.attached_files_items = widgets.VBox([])
        
        self.message = widgets.Label(value='')
        self.upload = widgets.Button(icon='plus', button_style='success')
        self.upload.on_click(self._upload_files)
        
        super(ImportTab, self).__init__([widgets.HBox([self.attach_files, self.upload]), self.message, self.attached_files_items])
    
    def remove_file(self, attached_file):
        self.attached_files.remove(attached_file)
        self.regenerate_attached_files()
    
    def regenerate_attached_files(self):
        self.attached_files_items.children = self.attached_files
    
    #only allow one file to be uploaded at a time
    def _attach_file_button_changed(self, change):
        self.message.value = ""
        
        #check new files against file of attached file objects
        for file in change["new"]:
            #check if the file matches any attached file objects file property
            if not any([file == attached_file.file["metadata"]["name"] for attached_file in self.attached_files]):
                self.attached_files.append(AttachedFile(change["new"][file], self))
        
        self.regenerate_attached_files()
        
    def _upload_files(self, b):
        global user_data
        global data
        for attached_file in self.attached_files:
            try:
                f=io.BytesIO(attached_file.file["content"])
                # f=open(attached_file.file, 'r')
                lines=[l.decode() for l in f.readlines()]
                ion_num=int(lines[7][lines[7].find("[")+1:lines[7].find("]")])
                ion = ion_lookup[ion_num]

                #this is needed to account for Target is a GAS line
                start_line = [n for n, l in enumerate(lines) if l.startswith(" ======= Target  Composition ========")][0]
                
                materials = []
                #second we need to find the material which is on row 15
                for line in lines[start_line + 4:]:
                    if line.startswith(" ========"):
                        break
                    else:
                        materials.append(line.lstrip().split(" ")[0])

                f=io.BytesIO(attached_file.file["content"]) #reread since we destroyed all info with the previous readlines
                df = pd.read_csv(f,
                                skiprows=start_line + 4 +len(materials) + 9,
                                delim_whitespace=True,
                                header=None,
                                skipfooter=13,
                                engine='python',
                                names=['Energy', "Energy Unit", 'dE/dx Elec.', 'dE/dx Nuclear', 'Range', 'Range Unit', 'Long', 'Long Unit', 'Lat', 'Lat Unit'])
                
                energy_map = {"GeV": 1e3, "MeV": 1, "keV": 1e-3, "eV": 1e-6} #convert all to MeV
                range_map = {"m": 1e3, "mm": 1, "um": 1e-3, "A": 1e-7} #convert all to mm
                #convert all energy units to MeV
                df["Energy Conversion"] = df["Energy Unit"].map(energy_map)
                df["Energy"] = df["Energy"].mul(df["Energy Conversion"], axis=0)

                #convert all range units to mm
                df["Range Conversion"] = df["Range Unit"].map(range_map)
                df["Range"] = df["Range"].mul(df["Range Conversion"], axis=0)

                #convert all long units to mm
                df["Long Conversion"] = df["Long Unit"].map(range_map)
                df["Long. Straggling"] = df["Long"].mul(df["Long Conversion"], axis=0)

                #convert all lat units to mm
                df["Lat Conversion"] = df["Lat Unit"].map(range_map)
                df["Lateral Straggling"] = df["Lat"].mul(df["Lat Conversion"], axis=0)

                df["dE/dx"] = df["dE/dx Elec."]

                #look at only std columns
                df_std = df[["Lateral Straggling", "Long. Straggling", "Range", "dE/dx", "Energy"]].sort_values(by="Energy")

                
                df_std["mat"] = "".join(materials)
                df_std["ion"] = ion
                
                #set the index to be the new columns
                df_std = df_std.set_index(["mat", "ion"])
                user_data = user_data.append(df_std)

                # global data
                data = data.append(user_data)
                #drop any duplicates
                data = data.drop_duplicates()
                # self.data = pd.concat([self.data, df_std], join='outer').sort_values(by=["mat", "ion", "Energy"]).drop_duplicates()
                
                self.message.value = "File uploaded successfully"
                self.message.style.text_color = "Green"
                
                attached_file.file_name.style.background = "#CDE3CE"
                
                data_check()

                user_data.to_csv(f"{user_data_folder}/user_data.csv")

            except Exception as e:
                self.message.value = "Error: processing file ''" + str(attached_file.file["metadata"]["name"]) + "'.'"
                self.message.style.text_color = "Red"
                attached_file.file_name.style.background = "#F4CBCB"
                
        #temp fix until we move this into a class of its own
        self.parent.update()

class SingleQuery(widgets.VBox):
    def __init__(self, parent):
        display(HTML('''
            <style>
                .top_row_dd2 select {
                    background-color: #FFE1E1 !important;
                }
                .calcd_value_generic input {
                    background-color: #FFD13F !important;
                    font-family: DejaVu Sans Mono, monospace;
                    font-weight: 950;
                }
            </style>
        '''))
        
        style = {'description_width': 'initial'}
        
        self.parent = parent
        self.mat_dropdown = widgets.Dropdown(
            options=data.index.unique(level="mat"), 
            description='Material:',
            disabled=False, 
            style = style
        )
        
        self.ion_dropdown = widgets.Dropdown(
            options=data.index.unique(level="ion"), 
            value="H", 
            description='Ion:', 
            disabled=False, 
            style = style
        )
        
        self.mat_dropdown.add_class("top_row_dd1")
        self.ion_dropdown.add_class("top_row_dd2")

        controls = widgets.HBox([self.mat_dropdown, self.ion_dropdown]).add_class("top_row")

        # Energy row
        energy_header = widgets.HBox([
            widgets.Label(value="Energy $\;MeV/u$",      layout=FixedSizedLayout()),
            widgets.Label(value="LET $\;MeV/(mg/cm^2)$", layout=FixedSizedLayout()),
            widgets.Label(value="Range $\;mm$",          layout=FixedSizedLayout()),
            widgets.Label(value="Z Straggling $\;mm$",   layout=FixedSizedLayout()),
            widgets.Label(value="XY Straggling $\;mm$",  layout=FixedSizedLayout()),
        ])
        
        self.energy_controls = widgets.HBox([
            widgets.BoundedFloatText(value=0,disabled=False, min=0, max=99999999, step=0.1, layout=FixedSizedLayout()).add_class("layer_input"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
        ])

        self.energy_controls.children[0].observe(self._energy_changed, names='value')

        # Range row
        range_header = widgets.HBox([
            widgets.Label(value="Range $\;mm$",          layout=FixedSizedLayout()),
            widgets.Label(value="Energy $\;MeV/u$",      layout=FixedSizedLayout()),
            widgets.Label(value="LET $\;MeV/(mg/cm^2)$", layout=FixedSizedLayout()),
            widgets.Label(value="Z Straggling $\;mm$",   layout=FixedSizedLayout()),
            widgets.Label(value="XY Straggling $\;mm$",  layout=FixedSizedLayout()),
        ])

        self.range_controls = widgets.HBox([
            widgets.BoundedFloatText(value=0,disabled=False, min=0, max=data["Energy"].max(), step=0.1, layout=FixedSizedLayout()).add_class("layer_input"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0,disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
        ])

        self.range_controls.children[0].observe(self._range_changed, names='value')

        # LET row
        let_header = widgets.HBox([
            widgets.Label(value="LET $\;MeV/(mg/cm^2)$", layout=FixedSizedLayout()),
            widgets.Label(value="Range $\;mm$",          layout=FixedSizedLayout()),
            widgets.Label(value="Energy $\;MeV/u$",      layout=FixedSizedLayout()),
            widgets.Label(value="Z Straggling $\;mm$",   layout=FixedSizedLayout()),
            widgets.Label(value="XY Straggling $\;mm$",  layout=FixedSizedLayout()),
        ])

        self.let_controls = widgets.HBox([
            widgets.BoundedFloatText(value=0, disabled=False, min=0, max=999999999, step=0.1, layout=FixedSizedLayout()).add_class("layer_input"),
            widgets.BoundedFloatText(value=0, disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0, disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0, disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
            widgets.BoundedFloatText(value=0, disabled=True, min=0, max=99999999, layout=FixedSizedLayout()).add_class("calcd_value_generic"),
        ])

        self.let_controls.children[0].observe(self._let_changed, names='value')

        super(SingleQuery, self).__init__([controls, energy_header, self.energy_controls, range_header, self.range_controls, let_header, self.let_controls])#, full_header, self.full_controls])

    def _energy_changed(self, change):
        result = lookup("Energy", self.mat_dropdown.value, self.ion_dropdown.value, self.energy_controls.children[0].value)
        self.energy_controls.children[1].value = result['let']
        self.energy_controls.children[2].value = result['range']
        self.energy_controls.children[3].value = result['long']
        self.energy_controls.children[4].value = result['lat']
    
    def _range_changed(self, change):
        result = lookup("Range", self.mat_dropdown.value, self.ion_dropdown.value, self.range_controls.children[0].value)
        self.range_controls.children[1].value = result['energy']
        self.range_controls.children[2].value = result['let']
        self.range_controls.children[3].value = result['long']
        self.range_controls.children[4].value = result['lat']

    def _let_changed(self, change):
        result = lookup("dE/dx", self.mat_dropdown.value, self.ion_dropdown.value, self.let_controls.children[0].value)
        self.let_controls.children[1].value = result['range']
        self.let_controls.children[2].value = result['energy']
        self.let_controls.children[3].value = result['long']
        self.let_controls.children[4].value = result['lat']
    
    def update(self):
        self.mat_dropdown.options = data.index.unique(level="mat")
        self.ion_dropdown.options = data.index.unique(level="ion")

class DataWidget(widgets.VBox):
    def __init__(self, parent):
        self.parent = parent
        self.axis_options = ["dE/dx", "Energy", "Lateral Straggling", "Long. Straggling", "Range"]
        self.cat_options = ["Ion", "Material"]

        self.x_dropdown = widgets.Dropdown(options=self.axis_options, value=self.axis_options[1], layout=ItemLayout())
        self.y_dropdown = widgets.Dropdown(options=self.axis_options, value=self.axis_options[0], layout=ItemLayout())

        self._recalc_plot_pivot()
        
        self.ion_dropdown = widgets.SelectMultiple(options=list(self.plot_pivot.Ion.unique()), value=("H", ), layout=ItemLayout())
        self.mat_dropdown = widgets.SelectMultiple(options=list(self.plot_pivot.Material.unique()), value=("Silicon", ), layout=ItemLayout())
        self.xlog_button = widgets.ToggleButton(value=True, layout=ItemLayout(), icon='check', button_style='success')
        self.ylog_button = widgets.ToggleButton(value=True, layout=ItemLayout(), icon='check', button_style='success')

        self.col_dropdown = widgets.Dropdown(options=self.cat_options, value="Material", layout=ItemLayout())
        self.hue_dropdown = widgets.Dropdown(options=self.cat_options, value="Ion", layout=ItemLayout())

        self.plot_type_dropdown = widgets.Dropdown(options=["line", "scatter"], value="line", layout=ItemLayout())
        self.update_button = widgets.Button(description="Update", button_style='success', layout=ItemLayout())
        
        self.out = widgets.Output()
        self.error_message = widgets.Label(value="", disabled=True, style={"text_color": "red"})

        # self.x_dropdown.observe(self._x_change, 'value')
        # self.y_dropdown.observe(self._y_change, 'value')
        self.xlog_button.observe(self._swap_button_state, 'value')
        self.ylog_button.observe(self._swap_button_state, 'value')
        self.update_button.on_click(self._interaction)

        children = [
            widgets.HBox([
                self._col("Ion", self.ion_dropdown),
                self._col("Material", self.mat_dropdown),
                widgets.VBox(
                    [
                        self._col("Column", self.col_dropdown),
                        self._col("Hue", self.hue_dropdown),
                    ]),
                widgets.VBox(
                    [
                        self._col("X", self.x_dropdown),
                        self._col("Y", self.y_dropdown),
                    ]),
                widgets.VBox(
                    [
                        widgets.HBox(
                        [
                            self._col("X-Log", self.xlog_button, ColLayout('70px', '70px')),
                            self._col("Y-Log", self.ylog_button, ColLayout('70px', '70px')),
                        ]),
                        self._col("Plot Kind", self.plot_type_dropdown),
                    ]),
                    self._col("Display", self.update_button),
            ]),
                self.error_message,
                self.out]
        
        self._interaction(None)

        super(DataWidget, self).__init__(children=children)
        
    def _recalc_plot_pivot(self):
        self.plot_pivot = data.reset_index().reset_index().melt(id_vars=['index', 'mat', 'ion'], var_name='type', value_name='value').pivot(columns=["type"], index=["ion", "mat", "index"], values="value").reset_index().drop(columns=["index"]).rename(columns={"ion": "Ion", "mat": "Material"}).astype({"Ion": "category", "Material": "category"})
    
    def _change_update_button(self, updating):
        if updating:
#             self.update_button.icon = "spin spinner"
            self.update_button.description = "Updating"
            self.update_button.disabled = True
        else:
#             self.update_button.icon = ""
            self.update_button.description = "Update"
            self.update_button.disabled = False
    
    def _interaction(self, b):
        self.error_message.layout.visibility = 'hidden'
        self._change_update_button(True)

        try:
            
            if self.col_dropdown.value == self.hue_dropdown.value:
                self.error_message.layout.visibility = 'visible'
                self.error_message.value = "Column variable can't match hue variable"
                self._change_update_button(False)
                return
            
            with self.out:
                clear_output(True)
                ion = self.ion_dropdown.value
                mat = self.mat_dropdown.value
                xlog = self.xlog_button.value
                ylog = self.ylog_button.value
                col = self.col_dropdown.value
                hue = self.hue_dropdown.value
                x = self.x_dropdown.value
                y = self.y_dropdown.value
                kind = self.plot_type_dropdown.value
                
                self._recalc_plot_pivot()
                
                # subset = plot_pivot.query(f"ion.isin(@ion) and mat.isin(@mat) and ({x} >= @x_range[0] and {x} <= @x_range[1]) and ({y} >= @y_range[0] and {y} <= @y_range[1])").copy()
                subset = self.plot_pivot.query("Ion.isin(@ion) and Material.isin(@mat)", engine='python').copy()
                subset["Ion"] = subset["Ion"].cat.remove_unused_categories()
                subset["Material"] = subset["Material"].cat.remove_unused_categories()
                
                if len(subset) <= 2:
                    self.error_message.layout.visibility = 'visible'
                    self.error_message.value = "No values in range"
                    self._change_update_button(False)
                    return
                
                cols = min(2, len(subset[col].unique()))
                
                #plot
                g = sns.relplot(data=subset, x=x, y=y, col=col, hue=hue, style=hue,
                                col_wrap=cols, facet_kws=dict(sharex=False, sharey=False),
                                height=5, legend='full', kind=kind)

                g.set(xscale=("log" if xlog else "linear"), yscale=("log" if ylog else "linear"))
                plt.show()
        except:
            pass
        finally:
            self._change_update_button(False)
    
    def _swap_button_state(self, b):
        button = b["owner"]
        if button.value:
            button.icon = 'check'
            button.button_style = 'success'
        else:
            button.icon = 'times'
            button.button_style = 'danger'

    # def _x_change(self, c):
    #     self.x_range_slider.min = plot_pivot[self.x_dropdown.value].min()
    #     self.x_range_slider.max = plot_pivot[self.x_dropdown.value].max()
    #     self.x_range_slider.value = (self.x_range_slider.min, self.x_range_slider.max)

    # def _y_change(self, c):
    #     self.y_range_slider.min = plot_pivot[self.y_dropdown.value].min()
    #     self.y_range_slider.max = plot_pivot[self.y_dropdown.value].max()
    #     self.y_range_slider.value = (self.y_range_slider.min, self.y_range_slider.max)

    def _col(self, title, item, layout=ColLayout()):
        return widgets.VBox([
            widgets.Label(
                value=title,
                layout=ItemLayout()),
            item
        ], layout=layout)
    
    def update(self):
        self.mat_dropdown.options = data.index.unique(level="mat")
        self.ion_dropdown.options = data.index.unique(level="ion")

class PySRIM(widgets.VBox):
    def __init__(self, parent):
        self.parent = parent
        style = {'description_width': 'initial', 'border-color': 'black'}
        self.mat_name = widgets.Text(placeholder='Material Name', description='Material Name:', disabled=False, style = style)
        self.run_data = widgets.Label()
        
        self.run_calc = widgets.Button(description="Run Calculation", disabled=False, button_style='success', layout=widgets.Layout(width="190px"))
        self.run_calc.on_click(self._calc_button_clicked)
        #self._calc_button_clicked()
        
        self.layers = []
        self.add_layer_at(0)
        
        super(PySRIM, self).__init__([self.mat_name, self.run_data, *self.layers, self.run_calc])
    
    def add_layer_at(self, index):
        layer = PySRIMLayerContainer(self, index)
        self.layers.insert(index, layer)
        self.update_children()

    def update_children(self):
        for layer_n, layer in enumerate(self.layers):
            layer.update_controls(layer_n)
            layer.update_children()
        self.children = tuple([self.mat_name, self.run_data, *self.layers, self.run_calc])
        #self.calculate()
        
    def remove_layer(self, index):
        self.layers.pop(index)
        self.update_children()
    
    def move_layer_up(self, index):
        self.layers.insert(index-1, self.layers.pop(index))
        self.update_children()
    
    def move_layer_down(self, index):
        self.layers.insert(index+1, self.layers.pop(index))
        self.update_children()
    
    def get_layer_count(self):
        return len(self.layers)
    
    def _calc_button_clicked(self, b):
        global user_data
        global data

        self.run_calc.disabled = True

        ions = [string.capwords(ion) for ion in data.index.unique(level="ion")]

        for i in ions:
            self.run_data.value = f"Calculating for {i}"
            try:
                ion = Ion(i, energy=2.5e9)
                toplayer = self.layers[0]
                widthslider = toplayer.children[-1]
                densityslider = toplayer.children[-2]
                layer_config = Layer({
                    string.capwords(layer.children[0].value): {
                        'stoich': layer.children[1].value,
                        'E_d': layer.children[2].value,
                        'lattice': layer.children[3].value,
                        'surface': layer.children[4].value
                    } for layer in toplayer.layers}, density=densityslider.value, width=widthslider.value)
#                 target = Target(srimlayers)
                srim = SR(layer_config, ion, output_type=5, energy_min=1e5)

                results = srim.runHeadless(os.environ['SRIM_PATH'])

                energy = 1e-3*results.data[0]/results.ion["A1"] #energy in MeV/u (energy in MeV / divided by the atmoic mass of the ion)
                let = results.data[1] #dE/dx in MeV/(mg/cm2)
                proj_range = 1e-3*results.data[3] #projected range in mm
                long_strag = 1e-3*results.data[4] #longitudinal straggling in mm
                lat_strag = 1e-3*results.data[5] #lateral straggling in mm
                
                temp_df = pd.DataFrame()
                temp_df["mat"] = [self.mat_name.value] * len(results.data[0])
                temp_df["ion"] = [ion.symbol.upper()] * len(results.data[0])
                temp_df["Energy"] = energy
                temp_df["dE/dx"] = let
                temp_df["Range"] = proj_range
                temp_df["Lateral Straggling"] = lat_strag
                temp_df["Long. Straggling"] = long_strag
                temp_df = temp_df.set_index(["mat", "ion"])

                user_data = user_data.append(temp_df)
                
            except Exception as e:
                self.run_data.value = f"Error calculating for {i}"
                pass
        #remove any duplicates
        
        user_data = user_data.drop_duplicates()

        data_check()
        user_data.to_csv(f"{user_data_folder}/user_data.csv", index=True)
        data = data.append(user_data)
        
        self.parent.update()

        self.run_data.value = "Done!"
        self.run_calc.disabled = False

class PySRIMLayerContainer(widgets.VBox):
    def __init__(self, parent_container: PySRIM, index: int):
        style = {
            'description_width': 'initial', 
            'border-color': 'black'
        }
        self.parent_container = parent_container

        self.add_button_child = widgets.Button(icon="plus", disabled=False, button_style='success', layout=FixedIconButtonLayout())
        self.add_button_child.on_click(self._add_button_child)
        
        self.layerheader = widgets.HBox([widgets.HTML(value=f"<b style='font-size:20px'>Layer {index}</b>")])
        
        self.headers = widgets.HBox([
            widgets.Text(value="Material",disabled=True, layout=FixedSizedLayout()),
            widgets.Text(value="Stoich",disabled=True, layout=FixedSizedLayout()),
            widgets.Text(value="E_d ",disabled=True, layout=FixedSizedLayout()),
            widgets.Text(value="Lattice",disabled=True, layout=FixedSizedLayout()),
            widgets.Text(value="Surface",disabled=True, layout=FixedSizedLayout()),
            self.add_button_child
        ])

        self.ion_density_slider = widgets.FloatSlider(
            value=1,
            min=0.001,
            max=100000,
            description="Density:",
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(min_width='390px', max_width='390px')
        )
        self.ion_width_slider = widgets.FloatSlider(
            value=100,
            min=0.001,
            max=100000,
            description="Width:",
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(min_width='390px', max_width='390px')
        )
        
        self.layers = []
        self.add_layer_at(0)
        
        super(PySRIMLayerContainer, self).__init__([self.layerheader, self.headers, *self.layers, self.ion_density_slider, self.ion_width_slider])
            
    def _add_button_child(self, b):
        self.add_layer_at(0)
        
    def add_layer_at(self, index):
        layer = PySRIMMultiQueryLayer(self, index)
        self.layers.insert(index, layer)
        self.update_children()
    
    def remove_layer(self, index):
        self.layers.pop(index)
        self.update_children()
    
    def move_layer_up(self, index):
        self.layers.insert(index-1, self.layers.pop(index))
        self.update_children()
    
    def move_layer_down(self, index):
        self.layers.insert(index+1, self.layers.pop(index))
        self.update_children()

    def update_children(self):
        for layer_n, layer in enumerate(self.layers):
            layer.update_controls(layer_n)
        self.children = tuple([self.layerheader, self.headers, *self.layers, self.ion_density_slider, self.ion_width_slider])
        #self.calculate()

    def get_layer_count(self):
        return len(self.layers)
    
    def update(self):
        for layer in self.layers:
            layer.update()
    
    def update_controls(self, index):
        self.index = index
        
        self.layerheader = widgets.HBox([widgets.HTML(value=f"<b style='font-size:20px'>Layer</b>")])

class PySRIMMultiQueryLayer(widgets.HBox):
    def __init__(self, parent_container: PySRIMLayerContainer, index: int):
        self.parent_container = parent_container
        
        ions = data.index.unique(level="ion")
        
        self.material = widgets.Dropdown(value=ions[0], options=ions, disabled=False, layout = FixedSizedLayout())
        self.stoich = widgets.BoundedFloatText(
            value = 1,
            min = 0,
            step = 0.1,
            redout = False,
            disabled = False,
            layout = FixedSizedLayout()
        )
        
        self.add_button = widgets.Button(icon="plus", disabled=False, button_style='success', layout=FixedIconButtonLayout())
        self.move_up_button = widgets.Button(icon="arrow-up",disabled=False,layout=FixedIconButtonLayout(), button_style='info')
        self.move_down_button = widgets.Button(icon="arrow-down",disabled=False,layout=FixedIconButtonLayout(), button_style='info')
        self.delete_button = widgets.Button(icon="trash", disabled=False, button_style='danger', layout=FixedIconButtonLayout())
        
        self.add_button.on_click(self._add_button_clicked)
        self.move_up_button.on_click(self._move_up_button_clicked)
        self.move_down_button.on_click(self._move_down_button_clicked)
        self.delete_button.on_click(self._delete_button_clicked)
        
        self.e_d = widgets.BoundedFloatText(
            value = 1,
            min = 0,
            step = 0.1,
            redout = False,
            disabled = False,
            layout = FixedSizedLayout()
        )
        self.lattice = widgets.BoundedFloatText(
            value = 1,
            min = 0,
            step = 0.1,
            redout = False,
            disabled = False,
            layout = FixedSizedLayout()
        )
        self.surface = widgets.BoundedFloatText(
            value = 1,
            min = 0,
            step = 0.1,
            redout = False,
            disabled = False,
            layout = FixedSizedLayout()
        )
        
        self.e_d.style.text_color = self.lattice.style.text_color = self.surface.style.text_color = 'black'
        
        c = [self.material, self.stoich, self.e_d, self.lattice, self.surface,
             self.add_button, self.move_up_button, self.move_down_button, self.delete_button]
        
        if self.parent_container.get_layer_count() <= 1:
            self.delete_button.disabled = True
        
        self.update_controls(index)
            
        super(PySRIMMultiQueryLayer, self).__init__(children=c)
    
    def _add_button_clicked(self, b):
        self.parent_container.add_layer_at(self.index+1)
    
    def _move_up_button_clicked(self, b):
        self.parent_container.move_layer_up(self.index)
        
    def _move_down_button_clicked(self, b):
        self.parent_container.move_layer_down(self.index)
    
    def _delete_button_clicked(self, b):
        self.parent_container.remove_layer(self.index)
    
    def update_controls(self, index):
        self.index = index
        
        self.move_down_button.disabled = self.delete_button.disabled = self.move_up_button.disabled = False
        
        if index == 0:
            self.move_up_button.disabled = True
        if index >= self.parent_container.get_layer_count()-1:
            self.move_down_button.disabled = True
        if self.parent_container.get_layer_count() <= 1:
            self.delete_button.disabled = True