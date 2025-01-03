type !'a hlist = |

type !'a element = |

module type S = sig
  type 'a u

  type !'a v

  type _ t =
    | [] : unit hlist t
    | ( :: ) : 'a t * 'b hlist t -> ('a -> 'b) hlist t
    | E : 'a u -> 'a v element t

  val length : 'a hlist t -> int

  type map_fn = {f: 'a. 'a u -> 'a u}

  val map : map_fn -> 'a t -> 'a t

  type 'a map2_fn = {f: 'b. 'b u -> 'a -> 'b u}

  val map2 : 'a map2_fn -> 'b t -> 'a list -> 'b t

  type 'b map2_acc_fn = {f: 'a. 'a u -> 'a u -> 'b -> 'a u * 'b}

  val map2_acc : 'b map2_acc_fn -> 'a t -> 'a t -> 'b -> 'a t * 'b

  type 'a fold_fn = {f: 'b. 'a -> 'b u -> 'a}

  val fold_left : 'a fold_fn -> 'a -> 'b t -> 'a

  type 'a map_to_list_fn = {f: 'b. 'b u -> 'a}

  val map_to_list : 'a map_to_list_fn -> 'b t -> 'a list

  type ('a, 'b) map2_to_list_fn = {f: 'c. 'c u -> 'a -> 'b}

  val map2_to_list : ('a, 'b) map2_to_list_fn -> 'c t -> 'a list -> 'b list

  type any = Any : 'a u -> any

  val to_any_list : 'a t -> any list

  val unwrap : 'a v element t -> 'a u
end

module Make (T : sig
  type 'a t

  type !'a tag
end) : S with type 'a u = 'a T.t and type 'a v = 'a T.tag = struct
  type 'a u = 'a T.t

  type 'a v = 'a T.tag

  type _ t =
    | [] : unit hlist t
    | ( :: ) : 'a t * 'b hlist t -> ('a -> 'b) hlist t
    | E : 'a u -> 'a v element t

  let rec length : type a. a hlist t -> int = function
    | [] ->
        0
    | _ :: xs ->
        1 + length xs

  type 'b map2_acc_fn = {f: 'a. 'a u -> 'a u -> 'b -> 'a u * 'b}

  let rec map2_acc : type a b. b map2_acc_fn -> a t -> a t -> b -> a t * b =
   fun f l1 l2 acc ->
    match (l1, l2) with
    | [], [] ->
        ([], acc)
    | x :: xs, y :: ys ->
        let hd, acc = map2_acc f x y acc in
        let tl, acc = map2_acc f xs ys acc in
        (hd :: tl, acc)
    | E x, E y ->
        let x, acc = f.f x y acc in
        (E x, acc)

  type map_fn = {f: 'a. 'a T.t -> 'a u}

  let rec map : type a. map_fn -> a t -> a t =
   fun f l ->
    match l with [] -> [] | x :: xs -> map f x :: map f xs | E x -> E (f.f x)

  type 'a map2_fn = {f: 'b. 'b T.t -> 'a -> 'b T.t}

  let map2 : type a b. a map2_fn -> b t -> a list -> b t =
   fun f l1 l2 ->
    let error = Invalid_argument "Different lengths" in
    let rec inner : type a b. a map2_fn -> b t -> a list -> b t * a list =
     fun f l ys ->
      match (l, ys) with
      | [], _ ->
          ([], ys)
      | x :: xs, _ ->
          let hd, ys = inner f x ys in
          let tl, ys = inner f xs ys in
          (hd :: tl, ys)
      | E x, y :: ys ->
          (E (f.f x y), ys)
      | E _, [] ->
          raise error
    in
    match inner f l1 l2 with l, [] -> l | _ -> raise error

  type 'a fold_fn = {f: 'b. 'a -> 'b T.t -> 'a}

  let rec fold_left : type a b. b fold_fn -> b -> a t -> b =
   fun f acc l ->
    match l with
    | [] ->
        acc
    | x :: xs ->
        let acc = fold_left f acc x in
        fold_left f acc xs
    | E x ->
        f.f acc x

  type 'a map_to_list_fn = {f: 'b. 'b T.t -> 'a}

  let map_to_list f l = fold_left {f= (fun acc x -> List.cons (f.f x) acc)} [] l

  type ('a, 'b) map2_to_list_fn = {f: 'c. 'c u -> 'a -> 'b}

  let map2_to_list f l1 l2 =
    fold_left
      {f= (fun (acc, xs) y -> (List.cons (f.f y (List.hd xs)) acc, List.tl xs))}
      ([], l2) l1
    |> fst

  type any = Any : 'a T.t -> any

  let rec to_any_list : type a. a t -> any list = function
    | x :: xs ->
        to_any_list x @ to_any_list xs
    | [] ->
        []
    | E x ->
        [Any x]

  let unwrap : type a. a v element t -> a u = function E x -> x
end

module Map (L1 : S) (L2 : S with type 'a v = 'a L1.v) = struct
  type map_fn = {f: 'a. 'a L1.u -> 'a L2.u}

  let rec map : type a. map_fn -> a L1.t -> a L2.t =
   fun f l ->
    match l with
    | [] ->
        L2.[]
    | x :: xs ->
        map f x :: map f xs
    | E x ->
        E (f.f x)
end
