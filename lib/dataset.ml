let shuffle seq () =
  let list = List.of_seq seq in
  let tagged = List.map (fun x -> (Stdlib.Random.bits (), x)) list in
  let sorted = List.sort (fun (x, _) (y, _) -> compare x y) tagged in
  let shuffled_list = List.map snd sorted in
  List.to_seq shuffled_list ()

let batch n (shape, seq) =
  let rec take n = function
    | xs when n = 0 ->
        Some ([], fun () -> xs)
    | Seq.Cons (x, xs) ->
        Option.bind (take (n - 1) (xs ())) (fun (xs, ys) -> Some (x @ xs, ys))
    | _ ->
        None
  in
  let rec inner n seq () =
    match take n @@ seq () with
    | Some (batch, rest) ->
        Seq.Cons (batch, inner n rest)
    | None ->
        Seq.Nil
  in
  (n :: shape, inner n seq)

let to_elements (shape, l) =
  Seq.map (fun l -> Runtime.HostValue.E (Ir.Tensor.of_list F32 shape l)) l

let epoch batch_size (shape, l) =
  let l = shuffle l in
  to_elements (batch batch_size (shape, l))

let fixed_iterations n batch_size dataset =
  Seq.take n @@ Seq.cycle @@ epoch batch_size dataset

let progress total seq =
  let open Progress in
  let iteration_printer =
    Printer.create ~to_string:(Format.sprintf "%2.2fit") ~string_len:9 ()
  in
  let bar =
    Line.(
      rate iteration_printer ++ spacer 2
      ++ bar ?style:(Some `UTF8) total
      ++ spacer 2 ++ count_to total ++ spacer 2 ++ eta total ++ spacer 2
      ++ elapsed () )
  in
  let info = Line.(rpad 13 string) in
  let bar = Multi.(line bar ++ line info) in
  let display = Display.start bar in
  let [set_progress; set_msg] = Display.reporters display in
  ( Seq.mapi
      (fun i x ->
        set_progress 1 ;
        if i = total - 1 then Display.finalise display ;
        x )
      seq
  , set_msg )
