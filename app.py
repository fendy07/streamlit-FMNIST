import urllib
import streamlit as st
from src.mnist_run import *
from src.resource import *

# main function to run the app
def main():
    st.set_page_config(page_title=MAIN_TITLE, layout='wide')
    sidebar_menu_ui()


def sidebar_menu_ui():
    st.sidebar.title(SIDEBAR_TITLE)
    app_mode = st.sidebar.selectbox("Select an option",
                                    [SIDEBAR_OPTION_1, SIDEBAR_OPTION_2, SIDEBAR_OPTION_3])
    if app_mode == SIDEBAR_OPTION_1:
        st.title(MAIN_TITLE)
        show_brief_data()
    elif app_mode == SIDEBAR_OPTION_2:
        run_app()
    elif app_mode == SIDEBAR_OPTION_3:
        st.code(get_file_content(APP_FILE_NAME))
    

def show_brief_data():
    train_images, train_labels, test_images, test_labels, class_names = load_data()
    col1, col2 = st.columns((1, 1))

    # First column: Show training data
    with col1:
        with st.expander("Show the first data."):
            show_data(train_images)
    # Second column: Show label data
    with col2:
        with st.expander("Show the labels."):
            show_data_labels(train_images, train_labels, class_names)

# Run the app
def run_app():
    st.title("Fashion MNIST Classification Using CNN Model")
    SELECTED_OPTIMIZER = optimizer_selector_ui()
    SELECTED_METRIC = metric_selector_ui()
    SELECTED_EPOCHS = epoch_selector_ui()
    if st.sidebar.button("Start Training"):
        run_mnist(selected_optimizer = SELECTED_OPTIMIZER, 
                  selected_metric = SELECTED_METRIC, 
                  selected_epochs = SELECTED_EPOCHS)
        selections = f'{SELECTED_OPTIMIZER}, {SELECTED_METRIC}, {SELECTED_EPOCHS}'
        if selections not in HYPER_PARAMS:
            HYPER_PARAMS.append(selections)
    
    # Check if there are selected hyperparameters
    if HYPER_PARAMS:
        selections = st.sidebar.multiselect("Choose the results and compare (Optimizer Metric Epoch)", HYPER_PARAMS)
        if selections:
            st.header("Comparison of Results")
            compare_plots(selections)

# Download a single file content
@st.cache_resource(show_spinner = False)
def get_file_content(path):
    url = f"https://raw.githubusercontent.com/fendy07/streamlit-FMNIST/main/{path}"
    response = urllib.request.urlopen(url)
    return response.read().decode('utf-8')

def optimizer_selector_ui():
    st.sidebar.markdown("# Optimizer")
    optimizer_name = st.sidebar.selectbox("Select Optimizer", OPTIMIZERS, 0)
    return optimizer_name

def metric_selector_ui():
    st.sidebar.markdown("# Metric")
    metric_name = st.sidebar.selectbox("Select Metric", list(METRICS.keys()), 0)
    return METRICS[metric_name]

def epoch_selector_ui():
    st.sidebar.markdown("# Epochs")
    epochs = st.sidebar.slider("Select Epochs", EPOCH_MIN, EPOCH_MAX, SELECTED_EPOCHS)
    return int(epochs)

def compare_plots(hyper_parameters):
    for key in hyper_parameters:
        if key in SAVE_IMAGES:
            parameters = key.split(" ")
            description = f"Optimizer: **{parameters[0]}**, Metric: **{parameters[1]}**, Epochs: **{parameters[2]}**"
            st.write(description)
            col1, col2 = st.columns((1, 1))
            with st.spinner("Loading..."):
                with col1:
                    st.pyplot(SAVE_IMAGES[key][0])
                with col2:
                    st.pyplot(SAVE_IMAGES[key][1])

# Hanya khusus digunakan untuk debug 
def compare_plot(hyper_parameters):
    col1, col2 = st.columns((1, 1))
    print(hyper_parameters)
    # if hyper_parameters in SAVE_IMAGES:
    parameters = hyper_parameters.split(" ")
    print(parameters)
    description = f"Optimizer: {parameters[0]}, Metric: {parameters[1]}, Epochs: {parameters[2]}"
    print(description)
    st.subheader(description)

    with col1:
        st.pyplot(SAVE_IMAGES[hyper_parameters][0])
    with col2:
        st.pyplot(SAVE_IMAGES[hyper_parameters][1])


if __name__ == "__main__":
    main()