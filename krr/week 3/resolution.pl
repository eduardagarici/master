use_module(library(lists)).

readData(L) :-
    see('C:\\master\\krr\\week 3\\data.txt'),
    read(L),
    read(end_of_file),
    seen.

writeOutput(X) :-
    tell('C:\\master\\krr\\week 3\\resolution_output.txt'),
    (X = 0 ->
        write('SATISFIABLE')
    ; write('UNSATISFIABLE')),
    told.


calcLen(List,(N,List)):- length(List,N).
delLen((_,List),(List)).

sortByLength(List,Sorted):- 
  maplist(calcLen,List,List1), 
  sort(0,@=<,List1, List2),
  maplist(delLen,List2,Sorted).

sortClauses([], []).
sortClauses([H|List], [H1|Out]) :-
  sort(H,H1),
  sortClauses(List, Out).   

resolution:-
    readData(List),
    sortClauses(List,L2),
    sortByLength(L2,L),
    do_resolve(L).

do_resolve(List):-
    (once(calculate_resolvent(List,Resolvent)) ->
    (Resolvent = [] -> 
        writeOutput(Resolvent);
        sortByLength([Resolvent|List],L),
        do_resolve(L));
    writeOutput(0)
    ).

calculate_resolvent(List, Resolvent):-
    member(C1, List),
    member(C2, List),
    C1 \== C2,
    clause_match(C1, C2, Result),
    (Result = false -> false;
     (member(Result, List)-> false;
      Resolvent = Result
     )    
    ).

clause_match(C1, C2, Result):-
    member(T1, C1),
    assign_value(T1,T2),
    (member(T2, C2) ->
        delete(T1, C1, L1),
        delete(T2, C2, L2),
        append(L1,L2, Resolvent),
        sort(Resolvent,Result)
     ; false
    ).

assign_value(n(X), X):-!.
assign_value(X,n(X)).


delete(H,[H|T],T).
delete(X,[H|T],[H|L]) :- delete(X,T,L),!.
