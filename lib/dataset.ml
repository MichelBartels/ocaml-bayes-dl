let shuffle list =
  let tagged = List.map (fun x -> (Stdlib.Random.bits (), x)) list in
  let sorted = List.sort compare tagged in
  List.map snd sorted

let batch n (shape, l) =
  let rec take n = function
    | xs when n = 0 ->
        Some ([], xs)
    | x :: xs ->
        Option.bind (take (n - 1) xs) (fun (xs, ys) -> Some (x @ xs, ys))
    | _ ->
        None
  in
  let rec loop n l =
    match take n l with Some (batch', xs) -> batch' :: loop n xs | None -> []
  in
  (n :: shape, loop n l)

let to_elements (shape, l) =
  List.map
    (fun l ->
      Runtime.Value.Host (Ir.Tensor.from_float_list ?shape:(Some shape) l) )
    l

let epoch batch_size (shape, l) () =
  let l = shuffle l in
  to_elements (batch batch_size (shape, l))

let list_fun_to_seq f =
  match f () with [] -> Seq.empty | x :: xs -> Seq.cons x @@ List.to_seq xs

let fixed_iterations n batch_size dataset =
  Seq.cycle (list_fun_to_seq @@ epoch batch_size dataset) |> Seq.take n
