open Dsl

let sgd ?(lr = 0.001) f =
  let open Parameters in
  let* params = params_for f in
  let [grad; loss] = Backpropagate.grad_and_value (to_fun f) params in
  return
    Ir.Var.List.[loss; [float_map2 (fun p g -> p -@ (g *.> lr)) params grad]]

let adamw ?(lr = 0.001) ?(betas = (0.9, 0.999)) ?(eps = 1e-8)
    ?(weight_decay = 0.0001) f =
  let open Parameters in
  let* params = params_for f in
  let [grad; loss] = Backpropagate.grad_and_value (to_fun f) params in
  let beta1, beta2 = betas in
  let* (E t) = new_param (E (Ir.Tensor.scalar_f32 1.)) in
  let* m = new_param (Runtime.HostValue.zeros (Ir.ValueType.of_vars params)) in
  let* v = new_param (Runtime.HostValue.zeros (Ir.ValueType.of_vars params)) in
  let params = float_map (fun p -> p -@ (p *.> (lr *. weight_decay))) params in
  let m =
    float_map2 (fun m g -> (beta1 *.< m) +@ ((1. -. beta1) *.< g)) m grad
  in
  let v =
    float_map2 (fun v g -> (beta2 *.< v) +@ ((1. -. beta2) *.< (g *@ g))) v grad
  in
  let m_hat =
    float_map
      (fun m -> m /@ broadcast_scalar_like (1. -.< (scalar_f32 beta1 **@ t)) m)
      m
  in
  let v_hat =
    float_map
      (fun v -> v /@ broadcast_scalar_like (1. -.< (scalar_f32 beta2 **@ t)) v)
      v
  in
  let adjustment =
    float_map2 (fun m v -> lr *.< m /@ (sqrt v +.> eps)) m_hat v_hat
  in
  let params = float_map2 (fun p a -> p -@ a) params adjustment in
  let t = t +.> 1. in
  return Ir.Var.List.[loss; [params; E t; m; v]]
