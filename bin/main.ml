open Iree_bindings
open Dsl

let () = Printexc.record_backtrace true

let sigmoid x = 1. /.< (1. +.< exp (-1. *.< x))

let reparametrize mean var shape =
  let eps = Random.normal_f32 shape in
  (* let eps = norm (zeros ([], F32)) (ones ([], F32)) shape in *)
  (* let eps = zeros (shape, F32) in *)
  mean +@ (eps *@ var)

let kl mean logvar var =
  sum [0]
  @@ Dsl.mean [0; 1]
  @@ (-0.5 *.< (1.0 +.< logvar -@ (mean *@ mean) -@ var))

let bayesian_parameter batch_size shape =
  let open Parameters in
  let* (E mean) = new_param (Runtime.HostValue.zeros (E (shape, F32))) in
  let* (E var) = new_param (E (Ir.Tensor.full F32 shape 0.001)) in
  let mean = Ir.Var.BroadcastInDim (mean, [batch_size]) in
  let var = Ir.Var.BroadcastInDim (var, [batch_size]) in
  let loss = kl mean (ln (var *.> 1000.)) (var *.> 1000.) in
  return @@ [E (reparametrize mean var (batch_size :: shape)); E loss]

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let shape = Ir.shape_of_var x in
  let batch_size = List.hd shape in
  let* [E w; E w_loss] = bayesian_parameter batch_size [in_dims; out_dims] in
  let* [E b; E b_loss] = bayesian_parameter batch_size [1; out_dims] in
  return @@ [E (activation (matmul x w +@ b)); E (w_loss +@ b_loss)]

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* [E z; E z_loss] = dense ~activation:tanh 784 512 x in
  let* [E mean; E mean_loss] = dense ~activation:Fun.id 512 embedding_dim z in
  let* [E logvar; E logvar_loss] =
    dense ~activation:Fun.id 512 embedding_dim z
  in
  return [E mean; E logvar; E (z_loss +@ mean_loss +@ logvar_loss)]

let decoder z =
  let open Parameters in
  let* [E z; E z_loss] = dense ~activation:tanh embedding_dim 512 z in
  let* [E z; E z'_loss] = dense 512 784 z in
  return [E z; E (z_loss +@ z'_loss)]

let mse x x' = sum [0; 1] @@ mean [0] ((x -@ x') *@ (x -@ x'))

let vae (Ir.Var.List.E x) =
  let open Parameters in
  let* [E mean'; E logvar; E encoder_loss] = encoder x in
  let shape = Ir.shape_of_var x in
  let batch_size = List.hd shape in
  let z = reparametrize mean' (exp logvar) [batch_size; 1; embedding_dim] in
  let* [E x'; E decoder_loss] = decoder z in
  let kl = kl mean' logvar @@ exp logvar in
  let mse = mse x x' in
  let loss = mse +@ kl +@ encoder_loss +@ decoder_loss in
  return @@ E loss

let optim = Optim.adamw ~lr:0.0001

let train x = optim (vae x)

let decode Ir.Var.List.[] =
  let open Parameters in
  let x = norm (zeros ([], F32)) (ones ([], F32)) [1; 1; embedding_dim] in
  let* [y; _] = decoder x in
  return y

module Device =
  ( val Pjrt_bindings.make
          "/home/michel/part-ii-project/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
    )

module Runtime = Runtime.Make (Device)
open Runtime

let batch_size = 128

let input_type = ([batch_size; 1; 784], Ir.Tensor.F32)

let train_step =
  let param_type = Parameters.param_type (E input_type) train in
  compile [param_type; E input_type]
  @@ fun [params; x] -> Parameters.to_fun (train x) params

let decode =
  let param_type = Parameters.param_type [] decode in
  compile param_type @@ fun params -> Parameters.to_fun (decode []) params

let train_step set_msg params x =
  (* let x = DeviceValue.of_host_value @@ E x in *)
  let [loss; params] = train_step [params; x] in
  let (E loss) = DeviceValue.to_host_value loss in
  set_msg @@ Printf.sprintf "Loss: %3.2f" @@ List.hd @@ Ir.Tensor.to_list loss ;
  params

let num_steps = 25000

let show_sample params () =
  let Runtime.DeviceValue.(inference_params :: _) = params in
  let Runtime.DeviceValue.[_; decoder_params] = inference_params in
  let y = decode ~collect:false [decoder_params] in
  let (E y) = DeviceValue.to_host_value y in
  Mnist.plot y

let train () =
  let params =
    Parameters.initial (E input_type) train |> DeviceValue.of_host_value
  in
  let dataset = Mnist.load_images Train in
  let dataset = Dataset.batch_tensors batch_size dataset in
  let dataset = Dataset.repeat ~total:num_steps dataset in
  let dataset =
    Dataset.map (fun x -> DeviceValue.of_host_value @@ E x) dataset
  in
  let generator = Dataset.to_seq ~num_workers:4 dataset in
  let generator, set_msg = Dataset.progress num_steps generator in
  let train_step = train_step set_msg in
  let rec loop i params generator =
    match Seq.uncons generator with
    | None ->
        params
    | Some (batch, generator) ->
        loop (i + 1) (train_step params batch) generator
  in
  loop 1 params generator

let _ =
  let params = train () in
  while true do
    show_sample params ()
  done
