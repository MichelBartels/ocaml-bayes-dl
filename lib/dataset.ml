type 'a t = {get: int -> 'a; length: int}

let make length get = {get; length}

let shuffle t =
  let perm = Array.init t.length (fun i -> (Stdlib.Random.bits (), i)) in
  Array.sort (fun (x, _) (y, _) -> compare x y) perm ;
  {get= (fun i -> t.get (snd perm.(i))); length= t.length}

let batch f n t =
  let m = t.length / n in
  { get=
      (fun i ->
        let batch = List.init n (fun j -> t.get @@ ((n * i) + j)) in
        f batch )
  ; length= m }

let batch_tensors n = batch Ir.Tensor.concatenate n

let map f t = {get= (fun i -> f (t.get i)); length= t.length}

let repeat t ~total =
  { get=
      (fun i ->
        let i = i mod t.length in
        t.get i )
  ; length= total }

let to_seq ?(num_workers = 1) ?(max_fetched = 16) t =
  let open Domainslib.Chan in
  let in' = make_unbounded () in
  let out = make_bounded max_fetched in
  let workers =
    List.init num_workers (fun _ ->
        Domain.spawn (fun () ->
            let rec loop () =
              match recv in' with
              | `Done ->
                  ()
              | `Work i ->
                  send out @@ t.get i ;
                  loop ()
            in
            loop () ) )
  in
  for i = 0 to t.length - 1 do
    send in' (`Work i)
  done ;
  Seq.init t.length
  @@ fun i ->
  let item = recv out in
  if i + 1 = t.length then (
    for _ = 0 to t.length - 1 do
      send in' `Done
    done ;
    List.iter (fun worker -> Domain.join worker) workers ) ;
  item

(* let shuffle seq () = *)
(*   let list = List.of_seq seq in *)
(*   let tagged = List.map (fun x -> (Stdlib.Random.bits (), x)) list in *)
(*   let sorted = List.sort (fun (x, _) (y, _) -> compare x y) tagged in *)
(*   let shuffled_list = List.map snd sorted in *)
(*   List.to_seq shuffled_list () *)

(* let batch n (shape, seq) = *)
(*   let rec take n = function *)
(*     | xs when n = 0 -> *)
(*         Some ([], fun () -> xs) *)
(*     | Seq.Cons (x, xs) -> *)
(*         Option.bind (take (n - 1) (xs ())) (fun (xs, ys) -> Some (x @ xs, ys)) *)
(*     | _ -> *)
(*         None *)
(*   in *)
(*   let rec inner n seq () = *)
(*     match take n @@ seq () with *)
(*     | Some (batch, rest) -> *)
(*         Seq.Cons (batch, inner n rest) *)
(*     | None -> *)
(*         Seq.Nil *)
(*   in *)
(*   (n :: shape, inner n seq) *)

(* let to_elements (shape, l) = *)
(*   Seq.map (fun l -> Runtime.HostValue.E (Ir.Tensor.of_list F32 shape l)) l *)

(* let epoch batch_size (shape, l) = *)
(*   let l = shuffle l in *)
(*   to_elements (batch batch_size (shape, l)) *)

(* let fixed_iterations n batch_size dataset = *)
(*   Seq.take n @@ Seq.cycle @@ epoch batch_size dataset *)

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
