
import streamlit as st
from classes import Layer, Conv2DParams, MaxPool2DParams
st.set_page_config(layout="wide")

if "layer_ids" not in st.session_state:  # active rows, in order
    st.session_state.layer_ids = set()
    st.session_state.layer_ids.add(0)
    st.session_state.layers = {0: {
        "id": "",
        "layer": "",
        "params": {}}}
if "next_id" not in st.session_state:  # monotonically‑growing id
    st.session_state.next_id = 1  # 0 already used above


if "layers_selected" not in st.session_state:
    st.session_state.layers_selected =  []

if "allowed_layers" not in st.session_state:
    st.session_state.allowed_layers = [
        "Input"
    ]

    
def add_layer():
    st.session_state.layer_ids.add(st.session_state.next_id)
    st.session_state.layers[st.session_state.next_id] =  {
        "id": "",
        "layer": "",
        "params": {}}
    st.session_state.next_id += 1
    


def delete_layer(layer_id: int):  # remove row + its widgets
    if len(st.session_state.layer_ids) == 1:
        return
    if layer_id in st.session_state.layer_ids:
        st.session_state.layer_ids.remove(layer_id)
    # wipe keys belonging to that row so they don’t linger
    for k in list(st.session_state.keys()):
        if k.endswith(f"_{layer_id}"):
            del st.session_state[k]


def duplicate_layer(layer_id: int):  # copy widgets → new row
    new_id = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.layer_ids.append(new_id)

    # clone any existing values
    suffixes = ["layer_select", "conv_param", "conv_value"]
    for suf in suffixes:
        old_key, new_key = f"{suf}_{layer_id}", f"{suf}_{new_id}"
        if old_key in st.session_state:
            st.session_state[new_key] = st.session_state[old_key]
            
def update_layer_selection(layer_id, type, param):
    match type:
        case "Input":
            st.session_state.layers[layer_id]["id"] = "input"
            st.session_state.layers[layer_id]["layer"] = "input"
        case "Conv2D":
            st.session_state.layers[layer_id]["layer"] = "conv2d"
        case "Relu":
            st.session_state.layers[layer_id]["layer"] = "relu"
        case "MaxPooling2D":
            st.session_state.layers[layer_id]["layer"] = "maxpooling2d"
        case "Flatten":
            st.session_state.layers[layer_id]["layer"] = "flatten"
        case "Dense":
            st.session_state.layers[layer_id]["layer"] = "dense"
        case "Dropout":
            st.session_state.layers[layer_id]["layer"] = "dropout"
    current_params = st.session_state.layers.get(layer_id, {}).get("params", {})
    value = list(param.values())[0]
    if value is not None and not str(value).startswith("--Select"):
        current_params[list(param.keys())[0]] = value
    st.session_state.layers[layer_id]["params"] = current_params


layer_col, visualizer_col = st.columns(2)

with st.sidebar:
    # show session all states as a dicionary
    st.write(st.session_state.layers)

with layer_col:
    st.button("Add Layer", on_click=add_layer)

    
    for layer_id in st.session_state.layer_ids:
        with st.container(border=True):
            # main widgets
            layer_col, param_type_col, param_val_col, dupe, delete = st.columns(
                [2, 2, 3, 1, 1]
            )
            if layer_id == 0:
                layer_name = layer_col.selectbox(
                    "Layer",
                    ["Input"],
                    key=f"layer_select_{layer_id}"  # tidier row
                )
            else:
                layer_name = layer_col.selectbox(
                    "Layer",
                    ["--Select a layer--",
                     "Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dropout"
                    ],
                    key=f"layer_select_{layer_id}"  # tidier row
                )
                
              
            # extra widgets for input
            if layer_name == "Input":
                input_shape = param_type_col.selectbox(
                    "Input Shape",
                    ["--Select a value--","1", "3"],
                    key=f"input_shape_{layer_id}",
                    on_change=lambda id=layer_id: update_layer_selection(id, "Input", {"out_channels": st.session_state[f"input_shape_{id}"]})
                )
                # st.session_state.layers_selected.append(Layer('Input'))

            # extra widgets for Conv2D
            if layer_name == "Conv2D":
                in_channels = st.session_state.layers[layer_id-1].get("params", {})["out_channels"]
                prev_layer_id = st.session_state.layers[layer_id-1]['id']
                current_params = st.session_state.layers.get(layer_id, {}).get('params', {})
                current_params['in_channels'] = in_channels
                current_params['layer'] = prev_layer_id
                st.session_state.layers[layer_id]["params"] = current_params
                params = ["out_channels", "filters", "kernel_size", "stride", "padding"]
                param_key_map = {                 # one fixed key per parameter type
                    "Filters": f"conv_out_channels_{layer_id}",
                    "Kernel Size": f"conv_kernel_size_{layer_id}",
                }
                
                conv_param = param_type_col.selectbox(
                    "Param",
                    params,
                    key=f"conv_param_{layer_id}"
                )

                param_val_col.text_input(
                    "Value",
                    value=st.session_state.layers.get(layer_id, {}).get('params', {}).get(conv_param, 0),
                    key=f"{conv_param}_value_{layer_id}",
                    on_change=lambda id=layer_id, conv_param=conv_param: update_layer_selection(id, "Conv2D", {conv_param: st.session_state[f"{conv_param}_value_{id}"]})
                )

                # slider_res = param_val_col.slider(
                #     "Value",
                #     min_value=1,
                #     max_value=512,
                #     value=st.session_state.layers.get(layer_id, {}).get('params', {}).get(conv_param, 0),
                #     key= f"{conv_param}_value_{layer_id}",
                #     on_change=lambda id=layer_id, conv_param=conv_param: update_layer_selection(id, "Conv2D", {conv_param: st.session_state[f"{conv_param}_value_{id}"]})
                # )
                

                
            else:
                # keep columns aligned when Conv2D isn’t selected
                param_type_col.empty()
                param_val_col.empty()

            # per‑row action buttons
            with dupe:
                st.button(
                    "📄",  # duplicate icon
                    key=f"dup_{layer_id}",
                    help="Duplicate this layer",
                    on_click=duplicate_layer,
                    args=(layer_id,),
                )
            with delete:
                st.button(
                    "🗑️",  # delete icon
                    key=f"del_{layer_id}",
                    help="Delete this layer",
                    on_click=delete_layer,
                    args=(layer_id,),
                )

with visualizer_col:
    st.header("Visualization")
    if "selected_layers" in st.session_state:
        for layer in st.session_state.selected_layers:
            st.write(f"Visualizing {layer}...")
            
