-- Run this in Supabase SQL Editor to create all tables

create table if not exists signals (
  id           integer primary key,
  timestamp    text,
  direction    text,
  entry        double precision,
  stop         double precision,
  target       double precision,
  confidence   integer,
  zone_type    text,
  rr           double precision,
  status       text,
  rationale    text
);

create table if not exists trades (
  id           bigint generated always as identity primary key,
  signal_id    integer,
  ticket       integer,
  entry_price  double precision,
  lots         double precision,
  status       text,
  close_price  double precision,
  pnl          double precision,
  r_multiple   double precision,
  timestamp    text
);

create table if not exists equity_history (
  id        bigint generated always as identity primary key,
  timestamp text,
  equity    double precision
);

create table if not exists status (
  id        integer primary key default 1,
  state     text,
  symbol    text,
  equity    double precision,
  spread    double precision,
  timestamp text
);
