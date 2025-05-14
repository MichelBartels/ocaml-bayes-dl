type t = Train | Test

let base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

let download_file filename =
  if not (Sys.file_exists "datasets") then
    Unix.mkdir "datasets" 0o755 ;
  let url = base_url ^ filename in
  let output_path = "datasets/" ^ filename in
  let cmd = Printf.sprintf "curl -o %s %s" output_path url in
  let _ = Sys.command cmd in
  output_path

let download_dataset () =
  let files = [
    "train-images-idx3-ubyte.gz";
    "train-labels-idx1-ubyte.gz";
    "t10k-images-idx3-ubyte.gz";
    "t10k-labels-idx1-ubyte.gz"
  ] in
  List.iter (fun file ->
    let path = "datasets/" ^ file in
    if not (Sys.file_exists path) then
      print_endline @@ "Downloading " ^ file ;
      let _ = download_file file in
      let cmd = Printf.sprintf "gunzip -f %s" path in
      ignore (Sys.command cmd)
  ) files

let images = function
  | Train ->
      download_dataset ();
      "datasets/train-images-idx3-ubyte"
  | Test ->
      download_dataset ();
      "datasets/t10k-images-idx3-ubyte"

let labels = function
  | Train ->
      download_dataset ();
      "datasets/train-labels-idx1-ubyte"
  | Test ->
      download_dataset ();
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
        |> Tensor.of_list F32 [1; 784]
      in
      img )

let plot (img : (Tensor.f32, float) Tensor.t) =
  let open Graphics in
  let scale = 50 in
  let w = 28 * scale in
  let h = 28 * scale in
  open_graph @@ " " ^ string_of_int w ^ "x" ^ string_of_int h ;
  set_window_title "MNIST" ;
  let open Tensor in
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

let save img path =
  let open Bigarray in
  let arr = Array1.create Int8_unsigned c_layout (28 * 28) in
  let open Tensor in
  for i = 0 to (28 * 28) - 1 do
    let v = get img [0; 0; i] in
    let colour = 255. *. v in
    let colour = int_of_float colour in
    let colour = max 0 (min 255 colour) in
    Array1.set arr i colour
  done ;
  Stb_image_write.png path ~w:28 ~h:28 ~c:1 arr
