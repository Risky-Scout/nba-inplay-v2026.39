#!/usr/bin/env python3
"""v2026.39 / engine 2026.29 regression checks."""
import json, os, tempfile
import numpy as np, pandas as pd
import kalshi_exec_model as XM
import fit_execution as FE
import grade_predictions as GP
import kalshi_nba_inplay_trader as T
import nba_markov_engine as MG
from kalshi_portfolio_qp import CorrelatedKellyQP
from execution_size_authority import size_bucket, highest_authorized_qty
from execution_hazard import cumulative_fill_probability, action_key
from rotation_hazard import RotationHazardModel, pathwise_blowout_probability
import residual_alpha as RA
import fit_residual_alpha as FRA
import risk_report as RR
import endgame_state as ES
import hiring_manager_audit as HMA
import nightly_reports as NR
from kalshi_ws import KalshiWSBook

fails=[]
def ck(name,cond,detail=""):
    print(("  [ok] " if cond else "  [FAIL] ")+name+(f" ({detail})" if detail else ""))
    if not cond:fails.append(name)

ck("versions",T.TRADER_VERSION=="v2026.39" and T.QUOTE_POLICY=="qp7" and FE.QUOTE_POLICY=="qp7" and GP.QUOTE_POLICY=="qp7" and MG.ENGINE_VERSION=="2026.29")
ck("valid numeric zero is preserved", MG.finite_or_default(0.0,5.5)==0.0 and MG.finite_or_default(np.nan,5.5)==5.5)
_epm=pd.DataFrame([{"team_id":1,"player_id":77,"player_name":"Semantic Test","p_mp_48":20.0,
    "p_mp_48_active":30.0,"p_active":.6,"p_played":.5,"p_usg":.2}])
_box=pd.DataFrame([{"team_id":1,"player_id":77,"p_mp_48":20.0,"p_mp_48_active":30.0,
    "p_active":.6,"p_played":.5,"p_pct_start":25.0,"pre_game_status":"Questionable","position":"G"}])
_sem,_=MG.build_dt_rosters_and_fallback_lines(_epm,_box,1)
_sem=_sem[0]
ck("D&T minutes and availability semantics stay distinct",
   _sem.p_mp==30.0 and _sem.p_mp_active==30.0 and _sem.p_mp_unconditional==20.0
   and abs(_sem.p_active-.6)<1e-12 and abs(_sem.p_played-.5)<1e-12
   and _sem.pre_game_status=="Questionable")

ck("drawdown includes starting equity", abs(RR._mdd([-10,-20])-20.0)<1e-12)
ck("bonus does not infer last-two-minute fouls from period total", not ES.in_bonus(2,4,90,None))
ck("explicit second last-two-minute foul triggers penalty", ES.in_bonus(2,4,90,2))
ck("overtime uses lower period foul threshold", ES.in_bonus(4,5,180,None))


# Audit estimands: ticker A is repeated 10x and bad; ticker B is one row and good.
_ad=pd.DataFrame({
    "ticker":["A"]*10+["B"],"game":["g1"]*10+["g2"],
    "timestamp":[f"2026-11-01T19:00:{i:02d}" for i in range(10)]+["2026-11-02T20:00:00"],
    "phase":["Q2"]*11,
    "p_dir":[.9]*10+[.9],"p_mkt_dir":[.5]*11,"y_dir":[0.0]*10+[1.0]
})
_views=HMA._paired_loss_views(_ad)
_md=_views["Market Loss Detail"]
_expected_market_brier=float(_md["model_brier"].mean())
_timestamp_brier=float(((_ad.p_dir-_ad.y_dir)**2).mean())
ck("equal-weighted market audit averages losses within ticker before equal weighting",
   abs(float(_views["Equal Weighted Markets"].iloc[0]["Model_Brier"])-_expected_market_brier)<1e-12
   and abs(_expected_market_brier-_timestamp_brier)>1e-3)
ck("phase snapshot audit keeps first forecast per ticker-phase",
   int(_views["Phase Snapshots"].iloc[0]["N"])==2)
ck("date-block audit preserves whole trading dates",
   int(_views["Date Block Summary"].iloc[0]["N_dates"])==2)


with tempfile.TemporaryDirectory() as _ld:
    pd.DataFrame([
      {"timestamp":"2026-11-11T10:00:00-05:00","trade_date_ny":"2026-11-11","mode":"LIVE","ticker":"T","side":"yes","contracts":2,"price_cents":40,"fee_usd":0},
      {"timestamp":"2026-11-11T10:10:00-05:00","trade_date_ny":"2026-11-11","mode":"LIVE","ticker":"T","side":"yes","contracts":-1,"price_cents":60,"fee_usd":0},
      {"timestamp":"2026-11-11T10:20:00-05:00","trade_date_ny":"2026-11-11","mode":"PAPER","ticker":"T","side":"yes","contracts":99,"price_cents":1,"fee_usd":0},
    ]).to_csv(os.path.join(_ld,"kalshi_fills_2026-11-11.csv"),index=False)
    _lp,_lo=NR.account_ledger_report(_ld,{"T":"yes"})
    ck("account ledger excludes PAPER rows",_lo.get("live_fill_rows")==2)
    ck("account ledger includes signed exit cash flow and settlement",abs(_lo.get("trading_equity_after_settlement_usd",999)-.8)<1e-12,str(_lo))

    pd.DataFrame([
      {"ticker":"E","requested_qty":2,"cond_edge_c":2.0,"effective_fill_fraction":.5}
    ]).to_csv(os.path.join(_ld,"kalshi_eligible_2026-11-11.csv"),index=False)
    _cp,_co=NR.capacity_report(_ld)
    ck("capacity report refuses linear bankroll extrapolation",
       len(_co)==3 and (~_co["bankroll_extrapolation_supported"].astype(bool)).all()
       and "bankroll" not in _co.columns)

for k in ("order_qty","qty_over_best_depth","qty_over_queue_ahead","log_order_qty"):
    ck("execution feature "+k,k in XM.EXEC_FEATURES)

x1=XM.exec_features("yes","PTS",3,45,50,46,4,ttl_s=8,qty_bid=20,qty_ask=20,queue_ahead=20,order_qty=1,best_depth=20)
x10=XM.exec_features("yes","PTS",3,45,50,46,4,ttl_s=8,qty_bid=20,qty_ask=20,queue_ahead=20,order_qty=10,best_depth=20)
ck("quantity changes feature vector",not np.allclose(x1,x10))
_n=len(XM.EXEC_FEATURES)
_x=np.ones(_n)
_m={"scaler":{"mean":[0.0]*_n,"sd":[1.0]*_n},
    "fill_any":[3.0]+[0.0]*(_n-1),
    "fill_fraction":[-3.0]+[0.0]*(_n-1),
    "mark_all":[0.0]*_n}
_qv=XM.quote_value(_m,_x,10.0,0.0)
ck("quote EV uses expected fill fraction, not probability of any fill",
   _qv["pf"]>0.9 and _qv["fill_fraction"]<0.1 and abs(_qv["qev"]-10.0*_qv["fill_fraction"])<1e-9,
   str(_qv))
ck("quote value exposes exact-request expected total value",
   abs(_qv["qev_total"]-_qv["expected_filled_qty"]*10.0)<1e-9)


with tempfile.TemporaryDirectory() as td:
    p=os.path.join(td,"old.json")
    json.dump({"quote_policy":"qp7","validated":True,"features":[x for x in XM.EXEC_FEATURES if x!="order_qty"],
               "scaler":{"mean":[0],"sd":[1]},"fill":[0],"mark_all":[0]},open(p,"w"))
    ck("old exec schema cannot load under qp7",XM.load_model(p,"qp7") is None)

q=pd.DataFrame([{"order_id":"a","requested_qty":10}])
ev=pd.DataFrame([{"order_id":"a","event":"fill","contracts":1,"reason":"","filled":1},
                 {"order_id":"a","event":"terminal","contracts":0,"reason":"CANCELLED_CONFIRMED: x","filled":1}])
lab=FE.label_quotes(q,ev)
ck("partial fill fraction",abs(float(lab.iloc[0]["fill_fraction"])-.1)<1e-12)
ck("partial fill remains any-fill",float(lab.iloc[0]["any_fill"])==1 and float(lab.iloc[0]["fully_filled"])==0)

ck("size buckets",[size_bucket(x) for x in [1,2,4,7]]==["QTY1","QTY2_3","QTY4_6","QTY7PLUS"])
auth={"buckets":{"QTY1":{"status":"PRODUCTION"},"QTY2_3":{"status":"PRODUCTION"},"QTY4_6":{"status":"PILOT"},"QTY7PLUS":{"status":"RESEARCH"}}}
ck("QTY1 cannot authorize QTY7PLUS",highest_authorized_qty(10,auth,True)==3)
ck("size fallback never increases",highest_authorized_qty(2,auth,True)<=2)
ck("exact action includes qty+TTL",action_key(50,1,8)!=action_key(50,2,8) and action_key(50,1,8)!=action_key(50,1,12))
ck("cumulative fill increases with horizon",cumulative_fill_probability([.1]*10)>cumulative_fill_probability([.1]*5))

# Queue progression: depletion moves us forward; additions after submission are behind; disappearance/reappearance resets.
_wb = KalshiWSBook(lambda: {})
_wb.want = {"Q"}
_wb._handle({"type":"orderbook_snapshot","sid":1,"seq":1,"msg":{
    "market_ticker":"Q","yes_dollars_fp":[["0.50","10"]],"no_dollars_fp":[["0.45","10"]]}})
_wb.start_queue_estimate("o","Q","yes",50,initial_depth=10)
_wb._handle({"type":"orderbook_delta","sid":1,"seq":2,"msg":{"market_ticker":"Q","price_dollars":"0.50","delta_fp":"-3","side":"yes"}})
_q1 = _wb.queue_estimate("o")
ck("queue ahead decreases on depth depletion", abs(_q1["queue_ahead_est"]-7.0)<1e-9, str(_q1))
_wb._handle({"type":"orderbook_delta","sid":1,"seq":3,"msg":{"market_ticker":"Q","price_dollars":"0.50","delta_fp":"5","side":"yes"}})
_q2 = _wb.queue_estimate("o")
ck("additions after submission are not assumed ahead", abs(_q2["queue_ahead_est"]-7.0)<1e-9, str(_q2))
_wb._handle({"type":"orderbook_delta","sid":1,"seq":4,"msg":{"market_ticker":"Q","price_dollars":"0.50","delta_fp":"-12","side":"yes"}})
_wb._handle({"type":"orderbook_delta","sid":1,"seq":5,"msg":{"market_ticker":"Q","price_dollars":"0.50","delta_fp":"4","side":"yes"}})
_q3 = _wb.queue_estimate("o")
ck("price-level reset invalidates prior queue priority", _q3["price_level_resets"]==1 and abs(_q3["queue_ahead_est"]-4.0)<1e-9, str(_q3))
_wf=KalshiWSBook(lambda:{})
_wf.want={"F"}
_wf._handle({"type":"orderbook_snapshot","sid":9,"seq":1,"msg":{
    "market_ticker":"F","yes_dollars_fp":[["0.505","10"]],"no_dollars_fp":[["0.49","10"]]}})
ck("fractional-cent WS book is rejected instead of rounded",
   "F" in _wf.fractional_price_tickers and _wf.best("F") is None)


rh=RotationHazardModel()
base=dict(player_id=1,team_id=1,role="starter",starter=True,completed_stints=1,period=4,seconds_remaining_in_period=360,
          fouls=2,projected_remaining_minutes=12,normal_window=True)
ck("long stint raises sub hazard",rh.sub_hazard_1s(current_stint_minutes=10,score_margin=4,**base)>rh.sub_hazard_1s(current_stint_minutes=6,score_margin=4,**base))
ck("Q4 blowout raises star sub hazard",rh.sub_hazard_1s(current_stint_minutes=7,score_margin=27,**base)>rh.sub_hazard_1s(current_stint_minutes=7,score_margin=4,**base))
ck("pathwise blowout probability responds to simulated score",pathwise_blowout_probability(27,300)>pathwise_blowout_probability(4,300))

prof=MG.DTPlayerSkillProfile(1,"P",1,3,78,210,100,32,.25,8,6,7,6,5,3,2,6,1,1,.65,.42,.36,.8,1,0,1)
eng=MG.InPlayMarkovEngine(MG.LiveInPlayState("g","g","Q1 12:00",1,2880,1,2,"H","A",0,0,"HOME",[],[],{}),[prof],[MG.DTPlayerSkillProfile(2,"Q",2,3,78,210,100,32,.25,8,6,7,6,5,3,2,6,1,1,.65,.42,.36,.8,1,0,1)],0,0,calib={1:{"PTS":1.2,"3PM":.9,"REB":1.1,"AST":1.15}})
cp=eng._calibrated_profile(prof)
ck("generative calibration changes attempt intensity",cp.p_fga_rim_100>prof.p_fga_rim_100 and cp.p_fg3a_100<prof.p_fg3a_100)
ck("generative calibration does not change shooting efficiency",cp.p_fgpct_rim==prof.p_fgpct_rim and cp.p_fg3pct==prof.p_fg3pct)
_role_state=MG.LiveInPlayState("g","g","Q2 6:00",2,1800,1,2,"H","A",50,48,"HOME",[1],[],{
    1:{"MIN":12.0,"_team_actions":40.0,"_team_minutes":60.0,"FTA":8,"TOV":5,"ORB":2}
},pbp_context=MG.LivePBPContext(live_fta={1:8},live_tov={1:5},live_orb={1:2},
    live_orb_opportunities={1:10},live_drb_opportunities={1:10}))
_role_eng=MG.InPlayMarkovEngine(_role_state,[prof],[MG.DTPlayerSkillProfile(2,"Q",2,3,78,210,100,32,.25,8,6,7,6,5,3,2,6,1,1,.65,.42,.36,.8,1,0,1)],0,0)
_role=_role_eng._calibrated_profile(prof)
ck("live role posterior updates FTA/TOV opportunity rates",_role.p_fta_100!=prof.p_fta_100 and _role.p_tov_100!=prof.p_tov_100)
ck("live role posterior never applies hot-hand FG adjustment",_role.p_fgpct_rim==prof.p_fgpct_rim and _role.p_fg3pct==prof.p_fg3pct)
# Same player production and same on-court exposure; extra team possessions/minutes occur while he is benched.
_s1=MG.LiveInPlayState("g","g","Q2 6:00",2,1800,1,2,"H","A",50,48,"HOME",[1],[],{
    1:{"MIN":10.0,"FTA":4,"TOV":2}
},pbp_context=MG.LivePBPContext(live_fta={1:4},live_tov={1:2}))
_e1=MG.InPlayMarkovEngine(_s1,[prof],[MG.DTPlayerSkillProfile(2,"Q",2,3,78,210,100,32,.25,8,6,7,6,5,3,2,6,1,1,.65,.42,.36,.8,1,0,1)],0,0)
_e1._aug_box={1:{"MIN":10.0,"FTA":4,"TOV":2,"_team_actions":20.0,"_team_minutes":50.0}}
_r1=_e1._calibrated_profile(prof)
_s2=MG.LiveInPlayState("g","g","Q2 6:00",2,1800,1,2,"H","A",50,48,"HOME",[],[],{
    1:{"MIN":10.0,"FTA":4,"TOV":2}
},pbp_context=MG.LivePBPContext(live_fta={1:4},live_tov={1:2}))
_e2=MG.InPlayMarkovEngine(_s2,[prof],[MG.DTPlayerSkillProfile(2,"Q",2,3,78,210,100,32,.25,8,6,7,6,5,3,2,6,1,1,.65,.42,.36,.8,1,0,1)],0,0)
_e2._aug_box={1:{"MIN":10.0,"FTA":4,"TOV":2,"_team_actions":40.0,"_team_minutes":100.0}}
_r2=_e2._calibrated_profile(prof)
ck("bench possessions do not dilute a player's live FTA/TOV rate",
   abs(_r1.p_fta_100-_r2.p_fta_100)<1e-12 and abs(_r1.p_tov_100-_r2.p_tov_100)<1e-12,
   f"{_r1.p_fta_100:.4f}/{_r2.p_fta_100:.4f}, {_r1.p_tov_100:.4f}/{_r2.p_tov_100:.4f}")

ck("legacy post calibration disabled by default",eng.legacy_post_calibration is False)

X=np.zeros((2,len(RA.FEATURES))); p=np.array([.3,.6])
ck("residual alpha uses fixed market offset",np.allclose(RA.predict({"beta":[0]*len(RA.FEATURES)},p,X),p))
_rdf=pd.DataFrame({
    "date":["2026-11-01"]*60+["2026-11-02"]*20,
    "game":["g1"]*30+["g2"]*30+["g3"]*20,
    "phase":["Q2"]*80,
    "p_market":np.full(80,.5),
    "y":np.r_[np.zeros(40),np.ones(40)],
    "const":np.ones(80),
    "x":np.r_[np.linspace(-1,1,60),np.linspace(100,120,20)]
})
_mdl=FRA.fit_model(_rdf.iloc[:60],["const","x"])
_pred=RA.predict(_mdl,_rdf.iloc[60:].p_market.values,_rdf.iloc[60:][["const","x"]].values)
ck("residual-alpha scaler is training-only and applied at predict time",
   abs(_mdl["scaler"]["mean"][1]-_rdf.iloc[:60].x.mean())<1e-12 and np.isfinite(_pred).all())
_w=FRA.evidence_weights(pd.DataFrame({"game":["a"]*10+["b"]*2,"phase":["Q2"]*12}))
ck("residual-alpha game/phase weighting prevents polling volume domination",
   abs(_w[:10].sum()-_w[10:].sum())<1e-9)


src=open("kalshi_nba_inplay_trader.py").read()
ck("live Kelly hard cap exists",'min(max(0.0, float(args.kelly)), 0.10) if args.live' in src)
ck("post-sizing size authority exists","highest_authorized_qty(_req, size_authority" in src)
ck("full-notional CVaR remains after size rescore","cvar_scale(orders, _others" in src)

if fails:
    print("FAILED:",fails); raise SystemExit(1)
print("ALL v2026.39 CHECKS PASSED")
