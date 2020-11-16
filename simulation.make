# -*- mode: Makefile -*-
#
# usage make -f THIS_FILE all
#
SEEDS := 0 1 2 3 4
ADM_KS := 2 3 4 5 10 20
MDA_KS := 2 3 4 5 10 20
PREFIX := data/sim.
PYTHON := nice -19 python
TREEGEN_OPT := --num_leaves=50 --fnum=100 --variance=5.0 --lambda=0.03 --nu=0.1 --merge_thres=0.95
ADM_OPT :=
MDA_OPT := --maxanneal=100 --gamma_shape=5.0 --gamma_scale=0.009023389356444755 --hmc_epsilon=0.025
EVAL_OPT := --min_ratio=0.1


define treegen
$(PREFIX)$(1).nexus :
	$(PYTHON) treegen.py --seed=$(1) $(TREEGEN_OPT) --nexus=$(PREFIX)$(1).nexus --langs=$(PREFIX)$(1).langs.json --flist=$(PREFIX)$(1).flist.json --tree=$(PREFIX)$(1).tree.json 2> $(PREFIX)$(1).log
$(PREFIX)$(1).langs.json : $(PREFIX)$(1).nexus
$(PREFIX)$(1).flist.json : $(PREFIX)$(1).nexus
$(PREFIX)$(1).tree.json : $(PREFIX)$(1).nexus

LANGS_LIST += $(PREFIX)$(1).langs.json
FLIST_LIST += $(PREFIX)$(1).flist.json
endef

$(foreach seed,$(SEEDS), \
  $(eval $(call treegen,$(seed))))



define admixture
$(PREFIX)$(1).adm.$(2).bins.json : $(PREFIX)$(1).langs.json $(PREFIX)$(1).flist.json
	$(PYTHON) admixture.py --seed=$(1) --K=$(2) $(ADM_OPT) --output=$(PREFIX)$(1).adm.$(2) --bins=$(PREFIX)$(1).adm.$(2).bins.json $(PREFIX)$(1).langs.json $(PREFIX)$(1).flist.json 2> $(PREFIX)$(1).adm.$(2).log

ADM_$(2)_LIST += $(PREFIX)$(1).adm.$(2).bins.json
ADM_LIST += $(PREFIX)$(1).adm.$(2).bins.json

$(PREFIX)$(1).adm.$(2).eval.json : $(PREFIX)$(1).tree.json $(PREFIX)$(1).adm.$(2).bins.json
	$(PYTHON) eval_clusters.py --model=adm $(EVAL_OPT) $(PREFIX)$(1).tree.json $(PREFIX)$(1).adm.$(2).bins.json > $(PREFIX)$(1).adm.$(2).eval.json

ADM_EVAL_LIST += $(PREFIX)$(1).adm.$(2).eval.json

endef

$(foreach seed,$(SEEDS), \
  $(foreach k,$(ADM_KS), \
    $(eval $(call admixture,$(seed),$(k)))))



define mda
$(PREFIX)$(1).mda.$(2).bins.json : $(PREFIX)$(1).langs.json $(PREFIX)$(1).flist.json
	$(PYTHON) mda.py --seed=$(1) --K=$(2) $(MDA_OPT) --output=$(PREFIX)$(1).mda.$(2) --bins=$(PREFIX)$(1).mda.$(2).bins.json $(PREFIX)$(1).langs.json $(PREFIX)$(1).flist.json 2> $(PREFIX)$(1).mda.$(2).log

MDA_$(2)_LIST += $(PREFIX)$(1).mda.$(2).bins.json
MDA_LIST += $(PREFIX)$(1).mda.$(2).bins.json

$(PREFIX)$(1).mda.$(2).eval.json : $(PREFIX)$(1).tree.json $(PREFIX)$(1).mda.$(2).bins.json
	$(PYTHON) eval_clusters.py --model=mda $(EVAL_OPT) $(PREFIX)$(1).tree.json $(PREFIX)$(1).mda.$(2).bins.json > $(PREFIX)$(1).mda.$(2).eval.json

MDA_EVAL_LIST += $(PREFIX)$(1).mda.$(2).eval.json

endef

$(foreach seed,$(SEEDS), \
  $(foreach k,$(MDA_KS), \
    $(eval $(call mda,$(seed),$(k)))))



langs : $(LANGS_LIST)
adm : $(ADM_LIST)
mda : $(MDA_LIST)
adm_eval : $(ADM_EVAL_LIST)
mda_eval : $(MDA_EVAL_LIST)

models : adm mda
eval : adm_eval mda_eval
all : eval
