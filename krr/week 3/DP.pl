use_module(library(lists)).

readData(L):-
    see('C:\\master\\krr\\week 3\\data.txt'),
    read(L),
    read(end_of_file),
    seen.

writeValues([]):-!.
writeValues([(A,V)|T]):-
    write((A,V)),nl,writeValues(T),!.

 writeOutput(X,Values) :-
    tell('C:\\master\\krr\\week 3\\dp_output.txt'),
    (X = [] ->
        (write('YES'),nl,
        writeValues(Values))
    ; write('NO')),
    told.   

davisPutnam:-
    readData(L),
    flatten(L,LF),
    setof((X,NR),(member(X,LF),countAtom(LF,X,NR)), Vars),
    eliminateNegatives(Vars,Vars,Vars2),
    sort(2,@=<,Vars2, SortedVars),  
    (do_resolve(L,SortedVars,[]) -> ! ;
     writeOutput(p,_)).

assignValue(_,true).
assignValue(_,false).

do_resolve(List,[(A,_)|T],Values):-
   calculateOp(List,A,Result,Values,NewValues),
   (Result = [] ->
        (writeOutput(Result,NewValues),!);
    (member([],Result) -> false;
        do_resolve(Result,T,NewValues)
    )
   ).
 
calculateOp(List,A,Result,Values,NewValues):-
    assignValue(A,V),
    append(Values,[(A,V)],NewValues),
    findall(R,(member(X,List),eliminate(X,A,V,R,R2), is_list(R2)),Res),
    sort(Res,Result).

eliminate([],_,_,[],[]):-!.
eliminate([n(A)|T],A,true,R,Result):-eliminate(T,A,true,R,Result),!.
eliminate([A|T],A,false,R,Result):-eliminate(T,A,false,R,Result),!.
eliminate([n(A)|_],A,false,_,p):-!.
eliminate([A|_],A,true,_,p):-!.
eliminate([H|T],A,V,[H|R],Result):-eliminate(T,A,V,R,Result),!.

countAtom([],_,0):-!.
countAtom([X|T],X,Y):- countAtom(T,X,Z), Y is 1+Z,!.
countAtom([n(X)|T],X,Y):-countAtom(T,X,Z),Y is 1+Z,!.
countAtom([_|T],X,Z):- countAtom(T,X,Z),!.

eliminateNegatives([],_,[]):-!.
eliminateNegatives([(n(X),_)|T],List,L):-member((X,_),List),eliminateNegatives(T,List,L),!.
eliminateNegatives([(n(X),Nr)|T],List,[(X,Nr)|L]):- eliminateNegatives(T,List,L),!.
eliminateNegatives([(X,Nr)|T],List,[(X,Nr)|L]):- eliminateNegatives(T,List,L),!.