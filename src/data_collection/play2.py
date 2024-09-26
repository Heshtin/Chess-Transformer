import json
def fen_to_vector(fen):
    fen_parts = fen.split(" ")
    rows = fen_parts[0].split("/")
    turn = fen_parts[1]
    if fen_parts[1] == "b":
        
        rows = [row.swapcase() for row in rows][::-1]
        
        fen_parts[2] = fen_parts[2].swapcase()
        #fen_parts[1] = 1
    # else:
    #     #fen_parts[1] = 0
    position = [0] #special token
    piece_dict = {" ":1, "p":2, "n":3, "b":4, "r":5, "q":6, "k":7, "P":8, "N":9, "B":10, "R":11, "Q":12, "K":13}
   
    for row in rows:
        for square in row:
            if square.isalpha():
                position.append(piece_dict[square])
            else:
                position.extend([1] * int(square))
    castling_rights = fen_parts[2]
    special_tokens = [0,0,0,0]
    for c in castling_rights:
        if c == "K":
            special_tokens[0] = 1
        elif c == "Q":
            special_tokens[1] = 1
        elif c == "k":
            special_tokens[2] = 1
        elif c == "q":
            special_tokens[3] = 1
    en_passant = fen_parts[3]
    if en_passant == "-":
        special_tokens.extend([0] * 9)
    else:
        file_index = ord(en_passant[0]) - 97
        special_tokens.extend([1] + [0] * file_index + [1] + [0] * (7 - file_index))
    

    json_position = json.dumps(position)
    json_special_tokens = json.dumps(special_tokens)

    return json_position, json_special_tokens, turn

# def fen_to_vector_2(fen):
#     fen_parts = fen.split(" ")
#     rows = fen_parts[0].split("/")
#     turn = fen_parts[1]
#     if fen_parts[1] == "b":
        
#         rows = [row.swapcase() for row in rows][::-1]
        
#         fen_parts[2] = fen_parts[2].swapcase()
#         #fen_parts[1] = 1
#     # else:
#     #     #fen_parts[1] = 0
#     position = [0] #special token
#     piece_dict = {" ":1, "p":2, "n":3, "b":4, "r":5, "q":6, "k":7, "P":8, "N":9, "B":10, "R":11, "Q":12, "K":13}
   
#     for row in rows:
#         for square in row:
#             if square.isalpha():
#                 position.append(piece_dict[square])
#             else:
#                 position.extend([1] * int(square))
#     castling_rights = fen_parts[2]
#     castling_rights_list = [0,0,0,0]
#     for c in castling_rights:
#         if c == "K":
#             castling_rights_list[0] = 1
#         elif c == "Q":
#             castling_rights_list[1] = 1
#         elif c == "k":
#             castling_rights_list[2] = 1
#         elif c == "q":
#             castling_rights_list[3] = 1
#     position.extend(castling_rights_list) 
#     en_passant = fen_parts[3]
#     if en_passant == "-":
#         position.extend([0] * 9)
#     else:
#         position.append(1)
#         file_index = ord(en_passant[0]) - 97
#         position.extend([0] * file_index)
#         position.append(1)
#         position.extend([0] * (7 - file_index))

#     json_position = json.dumps(position)

#     return position, turn

# def fen_to_vector_2(fen):
#     fen_parts = fen.split(" ")
#     rows = fen_parts[0].split("/")
#     turn = fen_parts[1]
#     if fen_parts[1] == "b":
        
#         rows = [row.swapcase() for row in rows][::-1]
        
#         fen_parts[2] = fen_parts[2].swapcase()
#         #fen_parts[1] = 1
#     # else:
#     #     #fen_parts[1] = 0
#     position="0," #special token
#     piece_dict = {" ":"1,", "p":"2,", "n":"3,", "b":"4,", "r":"5,", "q":"6,", "k":"7,", "P":"8,", "N":"9,", "B":"10,", "R":"11,", "Q":"12,", "K":"13,"}
    
#     for row in rows:
#         for square in row:
#             if square.isalpha():
#                 position+=piece_dict[square]
#             else:
#                 position+=int(square)*"1,"
#     castling_rights = fen_parts[2]
#     position += "1," if "K" in castling_rights else "0,"
#     position += "1," if "Q" in castling_rights else "0,"
#     position += "1," if "k" in castling_rights else "0,"
#     position += "1," if "q" in castling_rights else "0,"
#     en_passant = fen_parts[3]
#     if en_passant == "-":
#         position += "0," * 9
#     else:
#         position += "1,"
#         file_index = ord(en_passant[0]) - 97
#         position += "0," * file_index
#         position += "1,"
#         position += "0," * (7 - file_index)


#     return position[:-1], turn

def flip_uci(uci_move_string):
    out = ""
    out+=uci_move_string[0]
    out+=str(9 - int(uci_move_string[1]))
    out+=uci_move_string[2]
    out+=str(9 - int(uci_move_string[3]))
    return out











print(flip_uci("e1g1"))
print(fen_to_vector("2kr1bnr/p1p3pp/2nppq2/4p3/Pp1PP3/2N2N1P/1PPB1PP1/R2Q1RK1 b Kq - 0 1"))
#print(fen_to_vector_2("2kr1bnr/p1p3pp/2nppq2/4p3/Pp1PP3/2N2N1P/1PPB1PP1/R2Q1RK1 b Kq a3 0 1"))
print(fen_to_vector("2kr1bnr/p1p3pp/2nppq2/Pp2p3/3PP3/2N2N1P/1PPB1PP1/R2Q1RK1 b Qk a3 0 1"))
#print(fen_to_vector_2("2kr1bnr/p1p3pp/2nppq2/Pp2p3/3PP3/2N2N1P/1PPB1PP1/R2Q1RK1 w Qk b6 0 1"))





#out+=(chr(97 + 7 - (ord(uci_move_string[2]) - 97)))