open Iree_bindings
open Dsl

let () = Printexc.record_backtrace true

let sigmoid x = 1. /.< (1. +.< exp (-1. *.< x))

let bayesian_parameter batch_size shape =
  let open Parameters in
  let* (E mean) = new_param (Runtime.HostValue.zeros (E (shape, F32))) in
  let* (E var) = new_param (E (Ir.Tensor.full F32 shape @@ Float.log 0.001)) in
  let mean = Ir.Var.BroadcastInDim (mean, [batch_size]) in
  let var = Ir.Var.BroadcastInDim (exp var, [batch_size]) in
  (* return @@ mean *)
  return
  @@ Svi.sample
       ~prior:(Normal (zeros_like mean, ones_like mean *.> 0.001))
       ~guide:(Normal (mean, var))

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let shape = Ir.shape_of_var x in
  let batch_size = List.hd shape in
  let* w = bayesian_parameter batch_size [in_dims; out_dims] in
  let* b = bayesian_parameter batch_size [1; out_dims] in
  return @@ activation (matmul x w +@ b)

let embedding_dim = 16

let encoder x =
  let open Parameters in
  let* z = dense ~activation:tanh 784 512 x in
  let* mean = dense ~activation:Fun.id 512 embedding_dim z in
  let* logvar = dense ~activation:Fun.id 512 embedding_dim z in
  return (mean, logvar)

let decoder z =
  let open Parameters in
  let* z = dense ~activation:tanh embedding_dim 512 z in
  let* z = dense 512 784 z in
  return z

let vae x =
  let open Parameters in
  let* mean', logvar = encoder x in
  let z =
    Svi.sample
      ~prior:(Normal (zeros_like mean', ones_like mean'))
      ~guide:(Normal (mean', exp logvar))
  in
  let* x' = decoder z in
  return @@ Distribution.Normal (x', ones_like x')

let optim = Optim.adamw ~lr:0.000001

let train (Ir.Var.List.E x) = optim @@ Svi.elbo x @@ vae x

let decode Ir.Var.List.[] =
  let open Parameters in
  let x = norm (zeros ([], F32)) (ones ([], F32)) [1; 1; embedding_dim] in
  let* y = Svi.inference @@ decoder x in
  return @@ Ir.Var.List.E y

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
  set_msg @@ Printf.sprintf "Loss: %9.2f" @@ List.hd @@ Ir.Tensor.to_list loss ;
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
  let set_msg = print_endline in
  (* let generator, set_msg = Dataset.progress num_steps generator in *)
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
