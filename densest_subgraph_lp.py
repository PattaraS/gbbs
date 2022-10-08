#!/usr/bin/env python
# coding: utf-8

import sys
import gurobipy as gp
from gurobipy import GRB, quicksum

import pprint

adj_list = [
    (1,2),
    (1,3),
    (1,4),
    (2,3),
    (2,4),
    (3,4),
    (5,6),
    (6,7),
    (7,8),
    (8,9)
]

m = gp.Model()

grb_type = GRB.CONTINUOUS

vertex_vars = dict()
edge_vars = list()
for edge in adj_list:
    u = edge[0]
    v = edge[1]
    if u not in vertex_vars:
        vertex_vars[u] = m.addVar(lb=0, ub=1, name='vertex_{}'.format(u), vtype=grb_type)
    if v not in vertex_vars:
        vertex_vars[v] = m.addVar(lb=0, ub=1, name='vertex_{}'.format(v), vtype=grb_type)
    if u > v:
        u,v = v,u
    edge_var = m.addVar(lb=0, ub=1, name='edge_{}_{}'.format(u,v), vtype=grb_type)

    m.addConstr(edge_var <= vertex_vars[u])
    m.addConstr(edge_var <= vertex_vars[v])
    edge_vars.append(edge_var)

m.addConstr(quicksum(vertex_vars[u] for u in vertex_vars) <= 1)

m.setObjective(quicksum(e for e in edge_vars), GRB.MAXIMIZE)

m.optimize()

print('Density: {}'.format(m.objVal))

n = sum(1 for u in vertex_vars if vertex_vars[u].x > 0)
m = sum(1 for e in edge_vars if e.x > 0 )

print('n: {}, m: {}'.format(n,m))

