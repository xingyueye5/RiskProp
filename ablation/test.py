



model.load_state_dict(torch.load("baseline.pth"))
model.eval()

test_results = []
with torch.no_grad():
    for batch in test_loader:
        x, target = batch['inputs'].cuda(), batch['target']
        pred = model(x)

        for i in range(len(target)):
            test_results.append(dict(
                pred_score=pred[i].cpu(),
                target=target[i],
                video_id=batch['video_id'][i],
                frame_inds=batch['frame_inds'][i],
                abnormal_start_frame=batch['abnormal_start_frame'][i],
                accident_frame=batch['accident_frame'][i],
                is_val=batch['is_val'][i],
                is_test=batch['is_test'][i],
                dataset=batch['dataset'][i],
                frame_dir=batch['frame_dir'][i],
                filename_tmpl=batch['filename_tmpl'][i],
                type=batch['type'][i]
            ))

metric = AnticipationMetric(fpr_max=0.1, output_dir="outputs")
test_eval = metric.compute_metrics(test_results)
print(f"Test results: {test_eval}")
