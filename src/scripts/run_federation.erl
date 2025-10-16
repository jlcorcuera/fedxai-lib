-module(run_federation).
-export([start/0]).


start() ->
    case init:get_plain_arguments() of
        [Filename] ->
            run(Filename);
        _ ->
            usage(),
            erlang:halt(64)
    end.

run(Filename) ->
    DIRECTOR_NODE = list_to_atom(os:getenv("FEDLANG_DIRECTOR_NAME")),
    DIRECTOR_REF = {fedlang_director, DIRECTOR_NODE},
    case file:open(Filename, [read]) of
        {ok, IoDev} ->
            Cleaned = read_and_clean_lines(IoDev, []),
            file:close(IoDev),
            JsonStr = string:join(lists:reverse(Cleaned), ""),
            io:format("~s~n", [JsonStr]),
            gen_server:call(DIRECTOR_REF, {fl_start_str_run, JsonStr});
        {error, Reason} ->
            io:format("Error reading file: ~p~n", [Reason])
    end.

read_and_clean_lines(IoDev, Acc) ->
    case io:get_line(IoDev, "") of
        eof ->
            Acc;
        Line ->
            Cleaned = clean_line(Line),
            case Cleaned of
                ""   -> read_and_clean_lines(IoDev, Acc);
                Str  -> read_and_clean_lines(IoDev, [Str | Acc])
            end
    end.

clean_line(Line) ->
    Str0 = case is_binary(Line) of true -> binary_to_list(Line); false -> Line end,
    Str1 = string:replace(Str0, "\t", "", all),
    Str2 = string:trim(Str1, both),
    Str2.

usage() ->
    io:format("Usage: ./run_federation.sh <json_file>\n").
