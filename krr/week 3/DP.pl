use_module(library(lists)).

readData(L):-
    see('G:\\master\\krr\\week 3\\data.txt'),
    read(L),
    read(end_of_file),
    seen.

davisPutnam:-
    readData(L),
    flatten(L,LF),
    setof((X,NR),(member(X,LF),countAtom(LF,X,NR)), Vars),
    eliminateNegatives(Vars,Vars2),
    sort(2,@>=,Vars2, SortedVars),  
    do_resolve(L,SortedVars,[]).

assignValue(_,true).
assignValue(_,false).

do_resolve(List,[(A,_)|T],Values):-
   assignValue(A,V),
   once(calculateOp(List,A,V,Result)),



calculateOp(List,A,V,Result):-
    setOf(R,eliminate(X,A,V,R),member(X,List),Result).

eliminate([],_,_,[]):-!.
eliminate([not(A)|T],A,true,R):-eliminate(T,A,true,R),!.
eliminate([A|T],A,false,R):-eliminate(T,A,false,R),!.
eliminate([H|T],A,V,[H|R]):-eliminate(T,A,V,R),!.

countAtom([],X,0):-!.
countAtom([X|T],X,Y):- countAtom(T,X,Z), Y is 1+Z,!.
countAtom([n(X)|T],X,Y):-countAtom(T,X,Z),Y is 1+Z,!.
countAtom([_|T],X,Z):- countAtom(T,X,Z),!.

eliminateNegatives([],[]):-!.
eliminateNegatives([(n(X),NR)|T],L):- eliminateNegatives(T,L),!.
eliminateNegatives([(X,NR)|T],[(X,NR)|L]):- eliminateNegatives(T,L),!.