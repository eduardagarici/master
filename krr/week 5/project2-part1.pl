use_module(library(lists)).

program:-
    readKB(KB),
    performReasoning(KB).

performReasoning(KB):-
    displayMenu(KB,KBExtended),
    do_backchain(KBExtended,[p]),
    do_forwardchain(KBExtended,[p]),
    write('Stop?'),nl,
    read(Stop),
    (Stop \= 'stop' -> performReasoning(KB) ; !). 

readKB(L) :-
    see('C:\\master\\krr\\week 5\\kb.txt'),
    read(L),
    read(end_of_file),
    seen.

displayMenu(KB, KBExtended):-
    write('What is patient temperature?'),nl,
    read(Temp),
    write('For how many days has the patient been sick?'),nl,
    read(Sick),
    write('Has patient cough?'),nl,
    read(Cough),
    extendKB(KB,KBExtended,Temp,Sick,Cough).

extendKB(KB,KBExtended, Temp, Sick, Cough):-
    (Temp > 38 -> append(KB, [[t]], KB1) ; append(KB,[[n(t)]],KB1)),
    (Sick >= 2 -> append(KB1, [[s]], KB2) ; append(KB1, [[n(s)]], KB2)),
    (Cough = 'yes' -> append(KB2, [[c]], KBExtended) ; append(KB2, [[n(c)]], KBExtended)),!.

do_backchain(KB,Goals):-
    (backchain(KB,Goals)-> 
        write('Backchain: YES'),nl;
     write('Backchain: NO'),nl
    ).

backchain(_,[]):-!.
backchain(KB,[Atom|Tail]):-
    member(C, KB),
    member(Atom,C),
    checkAndDelete(Atom,C,List),
    append(List, Tail, Goals),
    backchain(KB, Goals),!.

do_forwardchain(KB,Goals):-
    (forwardchain(KB,Goals,[])->
        write('Forwardchain: YES'),nl;
     write('Forwardchain: NO'),nl
    ).

forwardchain(_,Goals,Solved):-all_from_first_in_second(Goals,Solved),!.
forwardchain(KB,Goals,Solved):-
    member(C, KB),
    check(C, Solved, Atom),
    append(Solved,[Atom], List),
    forwardchain(KB,Goals,List),!.

check([],_,_):-!.
check([H|T], Solved, H):- \+ member(H,Solved),check(T, Solved, H),!.
check([n(H)|T], Solved, Atom):-
    member(H,Solved),
    check(T,Solved,Atom),!.


checkAndDelete(_,[],[]):-!.
checkAndDelete(Atom,[Atom|Tail], List):-checkAndDelete(Atom,Tail,List),!.
checkAndDelete(Atom,[n(A)|Tail],[A|List]):-checkAndDelete(Atom,Tail,List),!.

all_from_first_in_second(List1, List2) :-
    forall(member(Element,List1), member(Element,List2)).