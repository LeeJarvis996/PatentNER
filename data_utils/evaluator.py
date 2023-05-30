class Evaluator(object):
    @staticmethod
    def check_match(ent_a, ent_b):
        return (ent_a.category == ent_b.category and
                max(ent_a.start_pos, ent_b.start_pos) < min(ent_a.end_pos, ent_b.end_pos))

    @staticmethod
    def count_intersects(ent_list_a, ent_list_b):
        num_hits = 0
        ent_list_b = ent_list_b.copy()
        for ent_a in ent_list_a:
            hit_ent = None
            for ent_b in ent_list_b:
                if Evaluator.check_match(ent_a, ent_b):
                    hit_ent = ent_b
                    break
            if hit_ent is not None:
                num_hits += 1
                ent_list_b.remove(hit_ent)
        return num_hits

    @staticmethod
    def f1_score(gt_docs, pred_docs):
        num_hits = 0
        num_preds = 0
        num_gts = 0
        for doc_id in gt_docs.doc_ids:
            gt_ents = gt_docs[doc_id].ents.ents # (T104615, product, (170597, 170599), 企业), (T113943, product, (170597, 170599), 企业), (T114307, product, (170597, 170599), 企业)
            pred_ents = pred_docs[doc_id].ents.ents  # [(T1, product, (17, 19), 医药), (T2, product, (23, 25), 医药), (T3, product, (76, 78), 管理), (T4, produc
            num_gts += len(gt_ents) # 209669
            num_preds += len(pred_ents) # 7773
            num_hits += Evaluator.count_intersects(pred_ents, gt_ents)
        p = num_hits / num_preds
        r = num_hits / num_gts
        f = 2 * p * r / (p + r)
        return f, p, r

