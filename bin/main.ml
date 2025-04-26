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

let save_sample params i =
  let Runtime.DeviceValue.(inference_params :: _) = params in
  let Runtime.DeviceValue.[_; decoder_params] = inference_params in
  let y = decode ~collect:false [decoder_params] in
  let (E y) = DeviceValue.to_host_value y in
  if not @@ Sys.file_exists "samples" then Unix.mkdir "samples" 0o755 ;
  Mnist.save y (Printf.sprintf "samples/sample_%d.png" i)

let train () =
  let params =
    Parameters.initial (E input_type) Vae.train |> DeviceValue.of_host_value
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
        if i mod 1000 = 0 then save_sample params i ;
        loop (i + 1) (train_step params batch) generator
  in
  loop 1 params generator

let _ = train ()

(* let _ = *)
(*   let params = train () in *)
(*   let _ = (params, show_sample) in *)
(*   while true do *)
(*     () (\* show_sample params () *\) *)
(*   done *)
