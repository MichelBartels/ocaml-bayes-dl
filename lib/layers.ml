open Dsl

let sigmoid x = 1.0 /.$ (1.0 +.$ exp (~-$x))

let bayesian_parameter batch_size shape std_prior =
  let open Parameters in
  let* (E mean) = new_param (Runtime.HostValue.zeros (E (shape, F32))) in
  let* (E log_std) =
    new_param (E (Tensor.full F32 shape @@ Float.log std_prior))
  in
  return
  @@ Svi.sample
       ~prior:(Normal (zeros_like mean, ones_like mean *$. std_prior))
       ~guide:(Normal (mean, exp log_std))
       ~batch_size ()

let dense_bayesian ?(activation = tanh) in_dims out_dims x =
  let open Parameters in
  let shape = Var.shape x in
  let batch_size = List.hd shape in
  let* w = bayesian_parameter batch_size [in_dims; out_dims] 0.01 in
  let* b = bayesian_parameter batch_size [1; out_dims] 0.01 in
  return @@ activation (matmul x w +$ b)

let dense ?(activation = tanh) in_dims out_dims x =
  let open Parameters in
  let shape = Var.shape x in
  let batch_size = List.hd shape in
  let* (E w) = new_param (E (Tensor.normal 0. 0.01 [in_dims; out_dims])) in
  let* (E b) = new_param (E (Tensor.normal 0. 0.01 [1; out_dims])) in
  let b = Var.BroadcastInDim (b, [batch_size]) in
  return @@ activation (matmul x w +$ b)