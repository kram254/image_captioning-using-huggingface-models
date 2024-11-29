def compute_score(self, option=None, verbose=0):
    n = self.n
    small = 1e-9
    tiny = 1e-15 ## so that if guess is 0 still return 0
    bleu_list = [[] for _ in range(n)]

    if self._score is not None:
        return self._score

    if option is None:
        option = "average" if len(self.crefs) == 1 else "closest"

    self._testlen = 0
    self._reflen = 0
    totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*n, 'correct':[0]*n}

        # for each sentence
    for comps in self.ctest:
        testlen = comps['testlen']
        self._testlen += testlen

        if self.special_reflen is None: ## need computation
            
            reflen = self._single_reflen(comps['reflen'], option, testlen)
        
          
        else:
            reflen = self.special_reflen

        self._reflen += reflen

        for key in ['guess','correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]

        # append per image bleu score
        bleu = 1.
        for k in range(n):
            bleu *= (float(comps['correct'][k]) + tiny) \
                /(float(comps['guess'][k]) + small)
            bleu_list[k].append(bleu ** (1./(k+1)))
            
        ratio = (testlen + tiny) / (reflen + small) ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleu_list[k][-1] *= math.exp(1 - 1/ratio)

        if verbose > 1:
            print(comps, reflen)

    totalcomps['reflen'] = self._reflen
    totalcomps['testlen'] = self._testlen

    bleus = []
    bleu = 1.
    for k in range(n):
        bleu *= float(totalcomps['correct'][k] + tiny) \
            / (totalcomps['guess'][k] + small)
        bleus.append(bleu ** (1./(k+1)))
        
    ratio = (self._testlen + tiny) / (self._reflen + small) ## N.B.: avoid zero division
    if ratio < 1:
        for k in range(n):
            bleus[k] *= math.exp(1 - 1/ratio)

    if verbose > 0:
        print(totalcomps)
        print("ratio:", ratio)

    self._score = bleus
