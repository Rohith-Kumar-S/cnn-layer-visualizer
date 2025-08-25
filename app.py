
import streamlit as st
import cv2
import numpy as np
from utils import ModelVisualizer
import copy

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

if "show_image" not in st.session_state:
    st.session_state.show_image = False

if "input_image" not in st.session_state:
    st.session_state.input_image = None

if "model_output" not in st.session_state:
    st.session_state.model_output = None

if "conv_count" not in st.session_state:
    st.session_state.conv_count = 0
if "relu_count" not in st.session_state:
    st.session_state.relu_count = 0
if "maxpool_count" not in st.session_state:
    st.session_state.maxpool_count = 0
if "flatten_count" not in st.session_state:
    st.session_state.flatten_count = 0
if "dense_count" not in st.session_state:
    st.session_state.dense_count = 0
if "dropout_count" not in st.session_state:
    st.session_state.dropout_count = 0
if "batchnorm_count" not in st.session_state:
    st.session_state.batchnorm_count = 0
if "layer_to_duplicate" not in st.session_state:
    st.session_state.layer_to_duplicate = None

if "model_string" not in st.session_state:
    st.session_state.model_string = ""

def add_layer():
    st.session_state.layer_ids.add(st.session_state.next_id)
    st.session_state.layers[st.session_state.next_id] =  {
        "id": "",
        "layer": "",
        "params": {}}
    st.session_state.next_id += 1
    


def delete_layer():  
    if len(st.session_state.layer_ids) == 1:
        return
    current_layer_name = st.session_state.layers[st.session_state.next_id-1]['layer']
    match current_layer_name:
        case "Conv2D":
            st.session_state.conv_count -= 1
        case "Relu":
            st.session_state.relu_count -= 1
        case "MaxPool2D":
            st.session_state.maxpool_count -= 1
        case "Flatten":
            st.session_state.flatten_count -= 1
        case "Dense":
            st.session_state.dense_count -= 1
        case "Dropout":
            st.session_state.dropout_count -= 1
        case "BatchNormalization":
            st.session_state.batchnorm_count -= 1
    st.session_state.layer_ids.remove(st.session_state.next_id-1)
    del st.session_state.layers[st.session_state.next_id-1]
    st.session_state.next_id -= 1


def duplicate_layer():  # copy widgets → new row
    if st.session_state.layer_to_duplicate is not None:
        new_id = st.session_state.next_id
        st.session_state.layers[st.session_state.next_id] =  {
            "id": "",
            "layer": "",
            "params": {}}
        prev_layer = st.session_state.layers[st.session_state.layer_to_duplicate]
        match prev_layer['layer']:
            case "Conv2D":
                st.session_state.layers[st.session_state.next_id]["id"] = f"conv2d_{st.session_state.conv_count}"
                st.session_state.layers[st.session_state.next_id]['params'] = copy.deepcopy(prev_layer['params'])
                st.session_state.layers[st.session_state.next_id]['params']['in_channels'] = st.session_state.layers[st.session_state.next_id-1]['params']['out_channels']
                st.session_state.layers[st.session_state.next_id]['params']['layer'] = st.session_state.layers[st.session_state.next_id-1]['id']
                st.session_state.conv_count += 1
            case "Relu":
                st.session_state.layers[st.session_state.next_id]["id"] = f"relu_{st.session_state.relu_count}"
                st.session_state.relu_count += 1
                st.session_state.layers[st.session_state.next_id]['params'] = copy.deepcopy(prev_layer['params'])
                st.session_state.layers[st.session_state.next_id]['params']['out_channels'] = st.session_state.layers[st.session_state.next_id-1]['params']['out_channels']
                st.session_state.layers[st.session_state.next_id]['params']['layer'] = st.session_state.layers[st.session_state.next_id-1]['id']
            case "MaxPool2D":
                st.session_state.layers[st.session_state.next_id]["id"] = f"maxpool2d_{st.session_state.maxpool_count}"
                st.session_state.maxpool_count += 1
                st.session_state.layers[st.session_state.next_id]['params'] = copy.deepcopy(prev_layer['params'])
                st.session_state.layers[st.session_state.next_id]['params']['out_channels'] = st.session_state.layers[st.session_state.next_id-1]['params']['out_channels']
                st.session_state.layers[st.session_state.next_id]['params']['layer'] = st.session_state.layers[st.session_state.next_id-1]['id']
        st.session_state.layers[st.session_state.next_id]['layer'] = prev_layer['layer']
        st.session_state.next_id += 1
        st.session_state.layer_ids.add(new_id)
        st.session_state.layer_to_duplicate = None
            
def update_layer_selection(layer_id, type, param):
    match type:
        case "Input":
            st.session_state.layers[layer_id]["id"] = "x"
            st.session_state.layers[layer_id]["layer"] = "Input"
        case "Conv2D":
            if st.session_state.layers[layer_id].get("id", "")=="":
                st.session_state.layers[layer_id]["id"] = f"conv2d_{st.session_state.conv_count}"
                st.session_state.layers[layer_id]["layer"] = "Conv2D"
                st.session_state.conv_count += 1
        case "Relu":
            if st.session_state.layers[layer_id].get("id", "")=="":
                st.session_state.layers[layer_id]["id"] = f"relu_{st.session_state.relu_count}"
                st.session_state.layers[layer_id]["layer"] = "Relu"
                st.session_state.relu_count += 1
        case "MaxPool2D":
            if st.session_state.layers[layer_id].get("id", "")=="":
                st.session_state.layers[layer_id]["id"] = f"maxpool2d_{st.session_state.maxpool_count}"
                st.session_state.layers[layer_id]["layer"] = "MaxPool2D"
                st.session_state.maxpool_count += 1
        case "Flatten":
            if st.session_state.layers[layer_id].get("id", "")=="":
                st.session_state.layers[layer_id]["id"] = f"flatten_{st.session_state.flatten_count}"
                st.session_state.layers[layer_id]["layer"] = "Flatten"
                st.session_state.flatten_count += 1
        case "Dense":
            if st.session_state.layers[layer_id].get("id", "")=="":
                st.session_state.layers[layer_id]["id"] = f"dense_{st.session_state.dense_count}"
                st.session_state.layers[layer_id]["layer"] = "Dense"
                st.session_state.dense_count += 1
        case "Dropout":
            if st.session_state.layers[layer_id].get("id", "")=="":
                st.session_state.layers[layer_id]["id"] = f"dropout_{st.session_state.dropout_count}"
                st.session_state.layers[layer_id]["layer"] = "Dropout"
                st.session_state.dropout_count += 1
    current_params = st.session_state.layers.get(layer_id, {}).get("params", {})
    value = list(param.values())[0]
    if value is not None and not str(value).startswith("--Select"):
        current_params[list(param.keys())[0]] = value
    st.session_state.layers[layer_id]["params"] = current_params

def update_layer_to_dupe():
    print("updating layer to dupe")
    print(st.session_state.layer_to_dupe)
    if st.session_state.layer_to_dupe!="--Select a layer--":
        st.session_state.layer_to_duplicate = st.session_state.layer_to_dupe

def run_model():
    layers = st.session_state.layers
    layers[st.session_state.next_id] = {}
    layers[st.session_state.next_id]["id"] = "Output"
    layers[st.session_state.next_id]["layer"] = "Output"
    layers[st.session_state.next_id]["params"] = {"layer": layers[st.session_state.next_id-1]['id']}
    st.session_state.model_output, st.session_state.model_string = ModelVisualizer('model').generate_model_code(st.session_state.input_image, list(layers.values()))

# with st.sidebar:
#     # show session all states as a dicionary
#     st.write(st.session_state.layers)

st.title("CNN Visualizer")

for layer_id in st.session_state.layer_ids:
    with st.container(border=True):
        # main widgets
        layer_col, param_type_col, param_val_col = st.columns(
            3, vertical_alignment="bottom"
        )
        if layer_id == 0:
            layer_name = layer_col.selectbox(
                f"Layer",
                ["Input"],
                key=f"layer_select_{layer_id}"  # tidier row
            )
        else:
            options = ["--Select a layer--",
                    "Conv2D", "Relu", "MaxPool2D", "Flatten", "Dense", "Dropout", "Output"
                ]
            layer_name = layer_col.selectbox(
                f"Layer {layer_id}",
                options,
                index = options.index(st.session_state.layers[layer_id]["layer"]) if st.session_state.layers[layer_id]["layer"] in options else 0,
                key=f"layer_select_{layer_id}"  # tidier row
            )

        match layer_name:
            case "Input":
                # extra widgets for inpu
                uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
                def toggle_image():
                    st.session_state.show_image = not st.session_state.show_image
                if uploaded_file is not None:
                    # Read file bytes and convert to NumPy array
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    
                    # Decode image with OpenCV
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR format
                    st.session_state.input_image = img
                if st.session_state.input_image is not None:
                    st.button("Show/Hide Image", on_click=toggle_image)
                
                if st.session_state.show_image:
                    st.image(st.session_state.input_image, caption="Uploaded BGR Image")
                input_shape = param_type_col.selectbox(
                    "Output Shape",
                    ["--Select a value--","1", "3"],
                    key=f"input_shape_{layer_id}",
                    on_change=lambda id=layer_id: update_layer_selection(id, "Input", {"out_channels": st.session_state[f"input_shape_{id}"]})
                )
                if input_shape=="3":
                    param_val_col.selectbox(
                        "Color space",
                        ["--Select a value--","RGB", "HSV"],
                        key=f"color_space_{layer_id}",
                        on_change=lambda id=layer_id: update_layer_selection(id, "Input", {"color_space": st.session_state[f"color_space_{id}"]})
                    )
            # st.session_state.layers_selected.append(Layer('Input'))

        # extra widgets for Conv2D
            case "Conv2D":
                in_channels = st.session_state.layers[layer_id-1].get("params", {})["out_channels"]
                prev_layer_id = st.session_state.layers[layer_id-1]['id']
                current_params = st.session_state.layers.get(layer_id, {}).get('params', {})
                current_params['in_channels'] = in_channels
                current_params['layer'] = prev_layer_id
                st.session_state.layers[layer_id]["params"] = current_params
                params = ["out_channels", "kernel_size", "stride", "padding"]
                
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
            
            case "Relu":
                in_channels = st.session_state.layers[layer_id-1].get("params", {})["out_channels"]
                prev_layer_id = st.session_state.layers[layer_id-1]['id']
                current_params = st.session_state.layers.get(layer_id, {}).get('params', {})
                current_params['out_channels'] = in_channels
                current_params['layer'] = prev_layer_id
                st.session_state.layers[layer_id]["params"] = current_params
                update_layer_selection(layer_id, "Relu", {"layer": prev_layer_id})
            
            case "MaxPool2D":
                in_channels = st.session_state.layers[layer_id-1].get("params", {})["out_channels"]
                prev_layer_id = st.session_state.layers[layer_id-1]['id']
                current_params = st.session_state.layers.get(layer_id, {}).get('params', {})
                current_params['out_channels'] = in_channels
                current_params['layer'] = prev_layer_id
                st.session_state.layers[layer_id]["params"] = current_params
                update_layer_selection(layer_id, "MaxPool2D", {"layer": prev_layer_id})
                
                params =  ["kernel_size", "stride", "padding"]
                
                maxpool2d_param = param_type_col.selectbox(
                    "Param",
                    params,
                    key=f"maxpool2d_param_{layer_id}"
                )

                param_val_col.text_input(
                    "Value",
                    value=st.session_state.layers.get(layer_id, {}).get('params', {}).get(maxpool2d_param, 0),
                    key=f"{maxpool2d_param}_value_{layer_id}",
                    on_change=lambda id=layer_id, maxpool2d_param=maxpool2d_param: update_layer_selection(id, "MaxPool2D", {maxpool2d_param: st.session_state[f"{maxpool2d_param}_value_{id}"]})
                )

            # case "Output":
            #     update_layer_selection(layer_id, "Output", {"layer": st.session_state.layers[layer_id-1]['id']})
            #     st.button("Run Model", on_click=lambda: run_model())
            #     if st.session_state.model_output is not None:
            #         print("bruh,", st.session_state.model_output.shape)
            #         idx = st.slider("Choose slice", 0, st.session_state.model_output.shape[-1]-1, 0)
            #         st.image(st.session_state.model_output[:,:,idx], caption="Output Image", clamp=True)

            case default:
                # keep columns aligned when Conv2D isn’t selected
                param_type_col.empty()
                param_val_col.empty()
with st.container(border=True):
    col1, col2 = st.columns([4,1], vertical_alignment="bottom")
    col1.selectbox(
                f"Layer",
                ["Output"],
                index=0
            )
    col2.button("Run Model", on_click=lambda: run_model())
    if st.session_state.model_output is not None:
        options = ["Output", "Model code", "Hide"]
        selection = st.segmented_control(
            "", options, default=options[0],selection_mode="single", label_visibility="collapsed"
        )
        if selection=="Output":
            print('hellop')
            idx = st.slider("Choose slice", 0, st.session_state.model_output.shape[-1]-1, 0)
            st.image(st.session_state.model_output[:,:,idx], caption=f"Output Image {'x'.join(map(str, st.session_state.model_output.shape))}", clamp=True,use_container_width=True)
        elif selection=="Model code":
            st.code(st.session_state.model_string, language="python")
        else:
            st.write("")

with st.container():
    col1, col2, col3, col4, col5 = st.columns([1.1,1.9, 1.4,1.2,1.5], vertical_alignment="bottom", gap="small")
    with col1:
        st.button("Add Layer", on_click=add_layer)
    with col2:
        options = ["--Select a layer--"]
        options.extend(list(st.session_state.layers.keys()))
        options.remove(0)  # can’t dupe input layer
        st.selectbox("Select Layer to Duplicate", options=options, key="layer_to_dupe", label_visibility="collapsed", index=st.session_state.layer_to_duplicate if st.session_state.layer_to_duplicate is not None else 0, on_change=lambda: update_layer_to_dupe())
    with col3:
        st.button(
                    "Duplicate layer",  # duplicate icon
                    key=f"duplicate_layer",
                    help="Duplicate last layer",
                    on_click=duplicate_layer
                )
    with col4:
        st.button(
                    "Delete layer",  # delete icon
                    key=f"delete_layer",
                    help="Delete this layer",
                    on_click=delete_layer
                )


            
