module type BACKEND = sig
  val try_load : unit -> (module Device_api.S) option

  val try_load_with_prompt : unit -> (module Device_api.S) option
end

let backends = [(module Pjrt : BACKEND); (module Iree : BACKEND)]

let load () =
  let backend =
    List.find_map
      (fun (module Backend : BACKEND) ->
        match Backend.try_load () with
        | Some device ->
            Some device
        | None ->
            None )
      backends
  in
  match backend with
  | Some backend ->
      backend
  | None -> (
      let backend =
        List.find_map
          (fun (module Backend : BACKEND) ->
            match Backend.try_load_with_prompt () with
            | Some device ->
                Some device
            | None ->
                None )
          backends
      in
      match backend with
      | Some backend ->
          backend
      | None ->
          failwith "No backend found. Please install IREE or PJRT bindings" )
