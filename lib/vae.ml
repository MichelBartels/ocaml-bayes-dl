open Dsl
open Layers

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
