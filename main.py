import streamlit as st
from msdes import *


def main():
    st.title("SDEs on a manifold")
    input_str = st.sidebar.text_input("Input variables (comma-separated)", value="x,y")
    param_str = st.sidebar.text_input("Parameters (comma-separated)", value="r")
    manifold_str = st.sidebar.text_input("Parameters (comma-separated)", value="sin(x)*cos(y),sin(x)*sin(y), cos(x)")
    dyn_str = st.sidebar.text_input("Dynamics (comma-separated)", value="0, 0")
    noise_scale = st.sidebar.slider("Noise scale", min_value=0., max_value=3., value=1., step=0.01)
    params = [sp.Symbol(s.strip()) for s in param_str.split(",")]
    # Create parameter sliders:
    # Create sliders for each parameter
    param_values = []
    init_param = [4.0, 3.0, 2., 1.]
    for i, param in enumerate(params):
        param_value = st.sidebar.slider(label=str(param), min_value=-10.0, max_value=10.0, value=init_param[i], step=0.01)
        param_values.append(param_value)
    # Store the current slider values in session state
    st.session_state.param_values = param_values
    inputs = sp.Matrix([sp.Symbol(s.strip()) for s in input_str.split(",")])
    dynamics1 = sp.Matrix([sp.sympify(s.strip()) for s in dyn_str.split(",")])
    manifold = sp.Matrix([sp.sympify(s.strip()) for s in manifold_str.split(",")])
    st.session_state.inputs = inputs
    st.session_state.params = params
    st.session_state.dynamics = dynamics1
    g = metric_tensor(manifold, inputs)
    mu, Sigma = coefficients(g, inputs)
    st.write("Metric tensor")
    st.write(g)
    st.write("Intrinsic BM drift:")
    st.write(mu)
    st.write("Intrinsic BM diffusion:")
    st.write(Sigma)

    f = sp.lambdify([inputs], manifold)

if __name__ == "__main__":
    main()
