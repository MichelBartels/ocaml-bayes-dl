open Iree_bindings
open Dsl

let () = Printexc.record_backtrace true

let sigmoid x = (tanh (x /.> 2.0) +.> 1.0) /.> 2.0

let batch_size = 128

let reparametrize mean var shape =
  let eps = Random.normal_f32 shape in
  (* let eps = norm (zeros ([], F32)) (ones ([], F32)) shape in *)
  (* let eps = zeros (Tensor_type (shape, F32)) in *)
  mean +@ (eps *@ var)

let kl mean logvar var =
  sum [0]
  @@ Dsl.mean [0; 1]
  @@ (-0.5 *.< (1.0 +.< logvar -@ (mean *@ mean) -@ var))

let bayesian_parameter shape =
  let open Parameters in
  let* (E mean) = new_param (Runtime.Value.zeros (E (shape, F32))) in
  let* (E var) = new_param (Host (Ir.Tensor.full F32 0.001 shape)) in
  let mean = Ir.Var.BroadcastInDim (mean, [batch_size]) in
  let var = Ir.Var.BroadcastInDim (var, [batch_size]) in
  let loss = kl mean (ln (var *.> 1000.)) (var *.> 1000.) in
  return @@ [E (reparametrize mean var (batch_size :: shape)); E loss]

let dense ?(activation = sigmoid) in_dims out_dims x =
  let open Parameters in
  let* [E w; E w_loss] = bayesian_parameter [in_dims; out_dims] in
  let* [E b; E b_loss] = bayesian_parameter [1; out_dims] in
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
  let z = reparametrize mean' (exp logvar) [batch_size; 1; embedding_dim] in
  let* [E x'; E decoder_loss] = decoder z in
  let kl = kl mean' logvar @@ exp logvar in
  let mse = mse x x' in
  return @@ E (mse +@ kl +@ encoder_loss +@ decoder_loss)
(* let optim = Optim.sgd 0.000001 *)

let optim = Optim.adamw ?lr:(Some 0.0001)

let train x = optim (vae x)

(* let train_compiled = *)
(*   let func = *)
(*     Parameters.create_func (Tensor_type ([batch_size; 1; 784], F32)) train *)
(*   in *)
(*   Ir.compile func *)

(* let reconstruct x = *)
(*   let open Parameters in *)
(*   let* [mean'; logvar; _] = encoder x in *)
(*   let z = reparametrize mean' logvar [batch_size; 1; embedding_dim] in *)
(*   let* x' = decoder z in *)
(*   return x' *)

(* let reconstruct_compiled = *)
(*   let func = *)
(*     Parameters.create_func *)
(*       (Tensor_type ([batch_size; 1; 784], F32)) *)
(*       (fun x -> *)
(*         let open Parameters in *)
(*         let* [y; _] = reconstruct x in *)
(*         return y ) *)
(*   in *)
(*   Ir.compile func *)

(* let decoder_compiled = *)
(*   let func = *)
(*     Parameters.create_func *)
(*       (Tensor_type ([batch_size; 1; embedding_dim], F32)) *)
(*       (fun x -> *)
(*         let open Parameters in *)
(*         let* [y; _] = decoder x in *)
(*         return y ) *)
(*   in *)
(*   Ir.compile func *)

(* let () = *)
(*   print_endline train_compiled ; *)
(*   Compile.compile train_compiled "train.vmfb" ; *)
(*   print_endline reconstruct_compiled ; *)
(*   Compile.compile reconstruct_compiled "reconstruct.vmfb" ; *)
(*   print_endline decoder_compiled ; *)
(*   Compile.compile decoder_compiled "decoder.vmfb" *)

(* let main = *)
(*   Backpropagate.diff Var (fun [a; b] -> *)
(*       matmul a (Ir.Var.BroadcastInDim (b, [4])) ) *)

(* let main = *)
(*   Ir.create_func *)
(*     (List_type [Tensor_type ([4; 2; 1], F32); Tensor_type ([1; 2], F32)]) *)
(*     main *)

(* let main_compiled = Ir.compile main *)

(* let () = *)
(*   print_endline "Compiled function:" ; *)
(*   print_endline main_compiled ; *)
(*   Runtime.simple_mul () ; *)
(*   print_endline @@ Compile.get_compiled_model main_compiled *)

open Runtime

let device = Device.make Cuda

let input_type = ([batch_size; 1; 784], Ir.F32)

let train_step =
  let param_type = Parameters.param_type (E input_type) train in
  Device.compile device [param_type; E input_type]
  @@ fun [params; x] -> Parameters.apply (train x) params

(* let print_tensor t = *)
(*   let (Host t) = Value.move_to_host t in *)
(*   Printf.printf "%s\n" (Ir.Tensor.to_string t) *)

let train_step params x =
  let x = Value.move_to_device device x in
  let [loss; params] = train_step [params; x] in
  ignore @@ Value.move_to_host loss ;
  (* set_msg @@ Printf.sprintf "Loss: %3.2f" @@ Ir.Tensor.to_scalar loss ; *)
  params

let warmup = 20

let num_steps = 100

(* let () = Gc.set {(Gc.get ()) with minor_heap_size= 262144 * 2048} *)

let train () =
  let params =
    Parameters.initial (E input_type) train |> Value.move_to_device device
  in
  let dataset = Mnist.load_images Train in
  let generator = Dataset.fixed_iterations num_steps batch_size dataset in
  let first = Seq.uncons generator |> Option.get |> fst in
  let generator = Seq.repeat first |> Seq.take (num_steps + warmup) in
  print_endline "Starting training" ;
  let times = ref [] in
  let start = Sys.time () in
  (* let generator, set_msg = Dataset.progress num_steps generator in *)
  let rec loop params generator =
    match Seq.uncons generator with
    | None ->
        params
    | Some (batch, generator) ->
        (* Gc.major () ; *)
        times := Sys.time () :: !times ;
        loop (train_step params batch) generator
  in
  ignore @@ loop params generator ;
  let times =
    List.fold_right
      (fun cur (prev, xs) -> (cur, (cur -. prev) :: xs))
      !times (start, [])
    |> snd
  in
  let times = List.rev times in
  let times = List.to_seq times |> Seq.drop warmup |> List.of_seq in
  let mean_time = List.fold_left ( +. ) start times /. float_of_int num_steps in
  let square_diff =
    List.map (fun x -> (x -. mean_time) ** 2.) times |> List.fold_left ( +. ) 0.
  in
  let std_dev = Float.sqrt @@ (square_diff /. float_of_int (num_steps - 1)) in
  Printf.printf "Mean time: %3.4f\n" mean_time ;
  Printf.printf "Standard deviation: %3.4f\n" std_dev

let _ = train ()

(* let main = *)
(*   Device.compile device (List_type []) *)
(*   @@ fun [] -> Random.normal_f32 [2] +@ Random.normal_f32 [2] *)

(* let inputs = Value.move_to_device device [] *)

(* let run () = *)
(*   let (Value.Host output) = main inputs |> Value.move_to_host in *)
(*   Printf.printf "Output: %s\n" (Ir.Tensor.to_string output) *)

(* let repeat n f = Seq.init n Fun.id |> Seq.iter (fun _ -> f ()) *)

(* let () = repeat 10 run *)
