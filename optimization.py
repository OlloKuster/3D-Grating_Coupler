import optax
import numpy as np

from postprocess import postprocess


def optimizer_fn(input_params, dJ_fn, num_steps=20, unfold=False, betas=np.array([1]), rmin=0.01):
    params = input_params[0]
    softplus_params = (input_params[1], input_params[2])

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    Js = []
    params_history = [params]

    prev_val = 0
    for beta in betas:
        if beta == betas[-1]:
            num_steps = 20

        for i in range(num_steps):
            value, gradient = dJ_fn(params, softplus_params=softplus_params, step_num=i, unfold=unfold, beta=beta,
                                    rmin=rmin)

            value = float(value)

            print(f"step = {i + 1}")
            print(f"\tvalue = {value:.4e}")
            print(f"\tgrad_norm = {np.linalg.norm(gradient):.4e}")

            updates, opt_state = optimizer.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = np.clip(params, 0, 1)

            Js.append(value)
            params_history.append(params)
            if np.abs(np.abs(prev_val - value) / value) < 1e-4:
                break
            prev_val = value

            if i % 10 == 0:
                postprocess(params, Js, f"grating_coupler_{beta}_{i}", beta, rmin)
    index_final = np.argmin(Js[-num_steps:])
    params_final = params_history[-num_steps:]
    return Js, params_final[index_final]

