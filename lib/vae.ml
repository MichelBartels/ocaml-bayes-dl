open Dsl

let sigmoid x = 1. /.< (1. +.< exp (-1. *.< x))

let bayesian_parameter batch_size shape std_prior =
  let open Parameters in
  let* (E mean) = new_param (Runtime.HostValue.zeros (E (shape, F32))) in
  let* (E log_std) =
    new_param (E (Ir.Tensor.full F32 shape @@ Float.log std_prior))
  in
  return
  @@ Svi.sample
       ~prior:(Normal (zeros_like mean, ones_like mean *.> std_prior))
       ~guide:(Normal (mean, exp log_std))
       ~batch_size ()

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let shape = Ir.shape_of_var x in
  let batch_size = List.hd shape in
  let* w =
    bayesian_parameter batch_size [in_dims; out_dims]
    @@ Float.sqrt (2. /. float_of_int in_dims)
  in
  let* b = bayesian_parameter batch_size [1; out_dims] 1. in
  return @@ activation (matmul x w +@ b)

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* z = dense ~activation:tanh 784 512 x in
  let* mean = dense ~activation:Fun.id 512 embedding_dim z in
  let* logstd = dense ~activation:Fun.id 512 embedding_dim z in
  return (mean, logstd)

let decoder z =
  let open Parameters in
  let* z = dense ~activation:tanh embedding_dim 512 z in
  let* z = dense 512 784 z in
  return z

let vae x =
  let open Parameters in
  let* mean', logstd = encoder x in
  let z =
    Svi.sample
      ~prior:(Normal (zeros_like mean', ones_like mean'))
      ~guide:(Normal (mean', exp logstd))
      ()
  in
  let* x' = decoder z in
  return @@ Distribution.Normal (x', ones_like x' *.> 0.1)

let optim = Optim.adamw ~lr:0.001

let train (Ir.Var.List.E x) = optim @@ Svi.elbo x @@ vae x

let decode Ir.Var.List.[] =
  let open Parameters in
  let x = norm (zeros ([], F32)) (ones ([], F32)) [1; 1; embedding_dim] in
  let* y = Svi.inference @@ decoder x in
  return @@ Ir.Var.List.E y
