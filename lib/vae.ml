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

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* z = dense 784 512 x in
  let* mean = dense ~activation:Fun.id 512 embedding_dim z in
  let* std = dense ~activation:exp 512 embedding_dim z in
  return (mean, std)

let decoder z =
  let open Parameters in
  let* z = dense_bayesian embedding_dim 512 z in
  let* z = dense_bayesian ~activation:Fun.id 512 784 z in
  return @@ sigmoid (z *$. 100.0)

let vae x =
  let open Parameters in
  let* mean', std = encoder x in
  let z =
    Svi.sample
      ~prior:(Normal (zeros_like mean', ones_like mean'))
      ~guide:(Normal (mean', std))
      ()
  in
  let* x' = decoder z in
  return @@ Distribution.Normal (x', ones_like x' *$. 0.01)

let optim = Optim.adamw ~lr:1e-3

let train (Var.List.E x) =
  optim @@ Svi.elbo_loss x @@ vae x

let decode Var.List.[] =
  let open Parameters in
  let x = Random.normal_f32 [1; 1; embedding_dim] in
  let* y = Svi.inference @@ decoder x in
  return @@ Var.List.E y

let reconstruct (Var.List.E x) =
  let open Parameters in
  let* y = Svi.inference @@ vae x in
  let y = Distribution.sample y None in
  return @@ Var.List.E y
