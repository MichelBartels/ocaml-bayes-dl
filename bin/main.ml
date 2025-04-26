open Iree_bindings

module Device =
  ( val Pjrt_bindings.make
          "/home/michel/part-ii-project/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
    )

module Runtime = Runtime.Make (Device)
open Runtime

let batch_size = 512

let input_type = ([batch_size; 1; 784], Ir.Tensor.F32)

let train_step =
  print_endline "getting param type" ;
  let param_type = Parameters.param_type (E input_type) Vae.train in
  print_endline "got param type" ;
  compile [param_type; E input_type]
  @@ fun [params; x] -> Parameters.to_fun (Vae.train x) params

let decode =
  let param_type = Parameters.param_type [] Vae.decode in
  compile param_type @@ fun params -> Parameters.to_fun (Vae.decode []) params

let reconstruct =
  let input_type = ([1; 1; 784], Ir.Tensor.F32) in
  let [param_type] = Parameters.param_type (E input_type) Vae.reconstruct in
  compile [param_type; E input_type]
  @@ fun [params; x] -> Parameters.to_fun (Vae.reconstruct x) [params]

let train_step set_msg params x =
  (* let x = DeviceValue.of_host_value @@ E x in *)
  let [loss; params] = train_step [params; x] in
  let (E loss) = DeviceValue.to_host_value loss in
  set_msg @@ Printf.sprintf "Loss: %15.9f" @@ List.hd @@ Ir.Tensor.to_list loss ;
  params

let num_steps = 25000

(* let show_sample params () = *)
(*   let Runtime.DeviceValue.(inference_params :: _) = params in *)
(*   let Runtime.DeviceValue.[_; decoder_params] = inference_params in *)
(*   let y = decode ~collect:false [decoder_params] in *)
(*   let (E y) = DeviceValue.to_host_value y in *)
(*   Mnist.plot y *)

let save_samples params x i =
  let Runtime.DeviceValue.(inference_params :: _) = params in
  let Runtime.DeviceValue.[_; decoder_params] = inference_params in
  let y = decode ~collect:false [decoder_params] in
  let (E y) = DeviceValue.to_host_value y in
  if not @@ Sys.file_exists "samples" then Unix.mkdir "samples" 0o755 ;
  Mnist.save y (Printf.sprintf "samples/sample_%d.png" i) ;
  let y = reconstruct ~collect:false [inference_params; x] in
  let (E y) = DeviceValue.to_host_value y in
  Mnist.save y (Printf.sprintf "samples/recon_%d.png" i)

let prepare_dataset dataset_type batch_size =
  let dataset = Mnist.load_images dataset_type in
  let dataset = Dataset.batch_tensors batch_size dataset in
  let dataset = Dataset.repeat ~total:num_steps dataset in
  Dataset.map (fun x -> DeviceValue.of_host_value @@ E x) dataset

let train () =
  let params =
    Parameters.initial (E input_type) Vae.train |> DeviceValue.of_host_value
  in
  let train_dataset = prepare_dataset Train batch_size in
  let sample_dataset = prepare_dataset Test 1 in
  let generator = Dataset.to_seq ~num_workers:4 train_dataset in
  let generator, set_msg = Dataset.progress num_steps generator in
  let sample_dataset = Dataset.to_seq ~num_workers:1 sample_dataset in
  let train_step = train_step set_msg in
  let rec loop i params train_set sample_set =
    match (Seq.uncons train_set, Seq.uncons sample_set) with
    | Some (batch, train_set), Some (sample, sample_set) ->
        if i mod 1000 = 0 then save_samples params sample i ;
        loop (i + 1) (train_step params batch) train_set sample_set
    | _ ->
        params
  in
  loop 1 params generator sample_dataset

let _ = train ()

(* let _ = *)
(*   let params = train () in *)
(*   let _ = (params, show_sample) in *)
(*   while true do *)
(*     () (\* show_sample params () *\) *)
(*   done *)
