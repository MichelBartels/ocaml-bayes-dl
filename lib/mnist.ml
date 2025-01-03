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
  let rec loop i =
    if i = n then []
    else
      let offset = 16 + (i * 784) in
      let img =
        List.init 784 (fun i ->
            float_of_int (String.get_uint8 str (offset + i)) /. 255. )
      in
      img :: loop (i + 1)
  in
  ([1; 784], List.to_seq @@ loop 0)
