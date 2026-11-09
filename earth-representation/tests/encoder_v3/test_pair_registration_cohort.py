import unittest
import json,tempfile
from pathlib import Path
from unittest.mock import patch
from encoder_v3.pair_cohort_lineage import main as audit_lineage_headers
from encoder_v3.pair_registration_cohort import select_pairs,check_stored_diagnostic


class RegistrationCohortChecks(unittest.TestCase):
    def test_full_qa_selection_preserves_all_times_and_input(self):
        members={k:dict(role='TRAIN',source_role='TRAIN',provenance='existing_native_TIF_with_QA') for k in ('s','t')}
        pairs=[dict(role='TRAIN',geometry_status='same_CRS',S2_id='s',Landsat_id='t',time_gap_seconds=gap) for gap in (60,1e7)]
        self.assertEqual(select_pairs(dict(pairs=pairs),members),pairs)
        self.assertEqual(len(pairs),2)
        self.assertEqual(select_pairs(dict(pairs=[dict(pairs[0],role='VAL')]),members),[])
        self.assertEqual(select_pairs(dict(pairs=[dict(pairs[0],geometry_status='common_projection_operator_required')]),members),[])
        members['t']['provenance']='native_lattice_cache_channels_only';self.assertEqual(select_pairs(dict(pairs=pairs),members),[])

    def test_role_mismatch_rejected(self):
        members={k:dict(role='TRAIN',source_role='TRAIN',provenance='existing_native_TIF_with_QA') for k in ('s','t')};members['s']['source_role']='TEST'
        with self.assertRaisesRegex(ValueError,'role/source'):select_pairs(dict(pairs=[dict(role='TRAIN',geometry_status='same_CRS',S2_id='s',Landsat_id='t')]),members)

    def test_running_cohort_rejected_before_imagery_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            index=Path(folder)/'index.json';index.write_text(json.dumps(dict(status='RUNNING_READ_ONLY_DIAGNOSTIC',rows=[])),encoding='utf-8')
            with patch('encoder_v3.pair_cohort_lineage.audit_member') as audit:
                with self.assertRaisesRegex(ValueError,'must finish'):audit_lineage_headers(index,Path(folder)/'output.json')
                audit.assert_not_called()

    def test_stored_diagnostic_requires_matching_recipe_and_sources(self):
        pair={'id':'explicit_pair'};study={'source_identities':{'members':'frozen'}};hashes={'pair_registration.py':'code'}
        prior=dict(status='REGISTRATION_DIAGNOSTIC_NOT_ADMISSION',pair=pair,source_identities=study['source_identities'],band_pairs=[dict(source_band='B03',target_band='SR_B3')],offsets_m=[0.],source_sha256=hashes)
        check_stored_diagnostic(prior,pair,study,['B03:SR_B3'],[0.],hashes)
        for field,value in [('pair',{}),('offsets_m',[1.]),('source_identities',{}),('source_sha256',{'pair_registration.py':'changed'})]:
            with self.assertRaises(ValueError):check_stored_diagnostic(dict(prior,**{field:value}),pair,study,['B03:SR_B3'],[0.],hashes)
