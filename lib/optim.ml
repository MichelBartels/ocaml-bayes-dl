open Dsl

let sgd lr f =
  let open Parameters in
  let* [grad; loss] = grad_and_value f in
  let* params = params in
  return [loss; float_map2 (fun p g -> p -@ (g *.> lr)) params grad]

let adamw ?(lr = 0.001) ?(betas = (0.9, 0.999)) ?(eps = 1e-08)
    ?(weight_decay = 0.01) f =
  let open Parameters in
  let* [grad; loss] = grad_and_value f in
  let* params = params in
  let beta1, beta2 = betas in
  let* t = new_param (Tensor_type ([], F32)) in
  let* m = new_param (Ir.ValueType.of_var params) in
  let* v = new_param (Ir.ValueType.of_var params) in
  let params =
    float_map2 (fun p g -> p -@ (g *.> (lr *. weight_decay))) params grad
  in
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
  return [loss; v; m; t; params]
