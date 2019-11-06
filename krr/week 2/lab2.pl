right_of(X, Y) :- X =:= Y+1.
left_of(X, Y) :- right_of(Y, X).

next_to(X, Y) :- right_of(X, Y).
next_to(X, Y) :- left_of(X, Y).


all_diff([X]).
all_diff([H|T]) :- \+member(H,T), all_diff(T).


Colors = [
    color(red),
    color(white),
    color(blue),
    color(yellow),
    color(green)
].

Nationalities = [
    nationality(british),
    nationality(swedish),
    nationality(danish),
    nationality(norwegian),
    nationality(german)
].
    
Drinks = [
    drink(milk),
    drink(beer),
    drink(tea),
    drink(water),
    drink(coffe)
].

Kiggaretes = [
    cigars(pallmall),
    cigars(winfield),
    cigars(marlboro),
    cigars(dunhill),
    cigars(rothmans)
].

Pets = [
    pet(bird),
    pet(dog),
    pet(horse),
    pet(cat),
    pet(fish)
].



question(Street, HasFish):-
    Street = [
        house(1, N1, C1, D1, K1, P1),
        house(2, N2, C2, D2, K2, P2),
        house(3, N3, C3, D3, K3, P3),
        house(4, N4, C4, D4, K4, P4),
        house(5, N5, C5, D5, K5, P5)],
    member(house(_,british,red,_,_,_), Street),
    member(house(A,norwegian,_,_,_,_), Street),
    member(house(B,_,blue,_,_,_), Street),
    next_to(A,B),
    member(house(C,_,green,_,_,_), Street),
    member(house(D,_,white,_,_,_), Street),
    left_of(C,D),
    member(house(_,_,green,coffe,_,_), Street),
    member(house(3,_,_,milk,_,_), Street),
    member(house(_,_,yellow,_,dunhill,_)),
    member(house(1,norwegian,_,_,_,_)),
    member(house(_,swedish,_,_,_,dog)),
    member(house(_,_,_,_,pallmall,bird)),
    member(house(E,_,_,_,marlboro,_)),
    member(house(F,_,_,_,_,cat)),
    next_to(E,F),
    member(house(_,_,_,beer,winfield,_)),
    member(house(G,_,_,_,_,horse)),
    member(house(H,_,_,_,dunhill,_)),
    next_to(G,H),
    member(house(_,german,_,_,rothmans,_)),
    member(house(I,_,_,_,marlboro,_)),
    member(house(J,_,_,water,_,_)),
    next_to(I,J),
    member(house(_,HasFish,_,_,_,fish), Street),
    member(nationality(N1),Nationalities),
    member(nationality(N2),Nationalities),
    member(nationality(N3),Nationalities),
    member(nationality(N4),Nationalities),
    member(nationality(N5),Nationalities),
    all_dif([N1,N2,N3,N4,N5]),
    member(color(C1),Colors),
    member(color(C2),Colors),
    member(color(C3),Colors),
    member(color(C4),Colors),
    member(color(C5),Colors),
    all_dif([C1,C2,C3,C4,C5]),
    member(drink(D1), Drinks),
    member(drink(D2), Drinks),
    member(drink(D3), Drinks),
    member(drink(D4), Drinks),
    member(drink(D5), Drinks),
    all_dif([D1,D2,D3,D4,D5]),
    member(cigars(K1),Kiggaretes),
    member(cigars(K2),Kiggaretes),
    member(cigars(K3),Kiggaretes),
    member(cigars(K4),Kiggaretes),
    member(cigars(K5),Kiggaretes),
    all_dif([K1,K2,K3,K4,K5]),
    member(pet(P1),Pets),
    member(pet(P2),Pets),
    member(pet(P3),Pets),
    member(pet(P4),Pets),
    member(pet(P5),Pets),
    all_dif([P1,P2,P3,P4,P5]).