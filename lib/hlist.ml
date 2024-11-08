module type S = sig
  type 'a u

  type !'a v

  type _ t = [] : unit t | ( :: ) : 'a u * 'b t -> ('a v -> 'b) t

  val length : 'a t -> int

  type map_fn = {f: 'a. 'a u -> 'a u}

  val map : map_fn -> 'a t -> 'a t

  type 'a map2_fn = {f: 'b. 'b u -> 'a -> 'b u}

  val map2 : 'a map2_fn -> 'b t -> 'a list -> 'b t

  type 'a fold_fn = {f: 'b. 'a -> 'b u -> 'a}

  val fold_left : 'a fold_fn -> 'a -> 'b t -> 'a

  type 'a map_to_list_fn = {f: 'b. 'b u -> 'a}

  val map_to_list : 'a map_to_list_fn -> 'b t -> 'a list

  type ('a, 'b) map2_to_list_fn = {f: 'c. 'c u -> 'a -> 'b}

  val map2_to_list : ('a, 'b) map2_to_list_fn -> 'c t -> 'a list -> 'b list

  type any = Any : 'a u -> any

  val to_any_list : 'a t -> any list
end

module Make (T : sig
  type 'a t

  type !'a tag
end) : S with type 'a u = 'a T.t and type 'a v = 'a T.tag = struct
  type 'a u = 'a T.t

  type 'a v = 'a T.tag

  type _ t = [] : unit t | ( :: ) : 'a T.t * 'b t -> ('a v -> 'b) t

  let rec length : type a. a t -> int = function
    | [] ->
        0
    | _ :: xs ->
        1 + length xs

  type map_fn = {f: 'a. 'a T.t -> 'a u}

  let rec map : type a. map_fn -> a t -> a t =
   fun f l -> match l with [] -> [] | x :: xs -> f.f x :: map f xs

  type 'a map2_fn = {f: 'b. 'b T.t -> 'a -> 'b T.t}

  let rec map2 : type a b. a map2_fn -> b t -> a list -> b t =
   fun f l1 l2 ->
    match (l1, l2) with
    | [], List.[] ->
        []
    | x :: xs, List.(y :: ys) ->
        f.f x y :: map2 f xs ys
    | _ ->
        failwith "map2: lists have different lengths"

  type 'a fold_fn = {f: 'b. 'a -> 'b T.t -> 'a}

  let rec fold_left : type a b. b fold_fn -> b -> a t -> b =
   fun f acc l ->
    match l with [] -> acc | x :: xs -> fold_left f (f.f acc x) xs

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
        Any x :: to_any_list xs
    | [] ->
        []
end

module Map (L1 : S) (L2 : S with type 'a v = 'a L1.v) = struct
  type map_fn = {f: 'a. 'a L1.u -> 'a L2.u}

  let rec map : type a. map_fn -> a L1.t -> a L2.t =
   fun f l -> match l with [] -> [] | x :: xs -> f.f x :: map f xs
end
