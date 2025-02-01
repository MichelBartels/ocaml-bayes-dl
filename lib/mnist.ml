type t = Train | Test

let images = function
  | Train ->
      "datasets/train-images-idx3-ubyte"
  | Test ->
      "datasets/t10k-images-idx3-ubyte"

let labels = function
  | Train ->
      "datasets/train-labels-idx1-ubyte"
  | Test ->
      "datasets/t10k-labels-idx1-ubyte"

let read path =
  let ch = open_in_bin path in
  let str = really_input_string ch (in_channel_length ch) in
  close_in ch ; str

let load_images t =
  let str = read (images t) in
  let magic = String.get_int32_be str 0 in
  assert (magic = 2051l) ;
  let n = String.get_int32_be str 4 |> Int32.to_int in
  Dataset.make n (fun i ->
      let offset = 16 + (i * 784) in
      let img =
        List.init 784 (fun i ->
            float_of_int (String.get_uint8 str (offset + i)) /. 255. )
        |> Ir.Tensor.of_list F32 [1; 784]
      in
      img )

let plot (img : (Ir.Tensor.f32, float) Ir.Tensor.t) =
  let open Graphics in
  let scale = 50 in
  let w = 28 * scale in
  let h = 28 * scale in
  open_graph @@ " " ^ string_of_int w ^ "x" ^ string_of_int h ;
  set_window_title "MNIST" ;
  let open Ir.Tensor in
  clear_graph () ;
  set_color black ;
  fill_rect 0 0 w h ;
  for i = 0 to 27 do
    for j = 0 to 27 do
      let x = i * scale in
      let y = j * scale in
      let v = get img [0; 0; (j * 28) + i] in
      let colour = int_of_float (255. *. (1. -. v)) in
      set_color (rgb colour colour colour) ;
      fill_rect x y scale scale
    done
  done ;
  print_endline "Press any key to continue" ;
  ignore @@ read_key () ;
  close_graph () ;
  print_endline "Continuing..."
