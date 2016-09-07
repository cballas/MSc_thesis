/*
# Author: Julien Hoachuck
# Copyright 2015, Julien Hoachuck, All rights reserved.
*/

#include <stdio.h>
#include <string.h>
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "luaT.h"
#include "TH/TH.h"

void computeCost(char** arr_a, float* arr_b, int paramSize, float *result)
{

    //printf("Entered the C file...\n");
    lua_State *L = luaL_newstate();
    luaL_openlibs( L );

    if (luaL_loadfile(L, "model.lua") || lua_pcall(L, 0, 0, 0))
    {
        printf("error: %s \n", lua_tostring(L, -1));
    } // lua_pcall(L,0,0,0) is necessary as a priming run? won't work if not there
    

    lua_getglobal(L, "trainHyper"); // tell what function to run
    if(!lua_isfunction(L,-1))
    {
        lua_pop(L,1);
    }
    int paramS = ;
    // Creating the table
   
    lua_newtable(L);
    int paramsIter;
    for(paramsIter = 0; paramsIter <= (paramSize-1); paramsIter++)
    {
        printf("%s\n", arr_a[paramsIter]);
        lua_pushstring(L,arr_a[paramsIter]);
        lua_pushnumber(L, arr_b[paramsIter]);
        lua_settable(L,-3);
    }

    // call the lua function in model.lua -> trainHyper(table)
    if (lua_pcall(L, 1, 1, 0) != 0)
    {
        printf("error running function `trainHyper': %s \n", lua_tostring(L, -1));
    }


    (*result) = lua_tonumber(L, -1);

    return;
}
