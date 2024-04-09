import argparse
import csv
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from tools.stats.upload_stats_lib import download_s3_artifacts, unzip, upload_to_rockset


ARTIFACTS = [
    "test-reports",
]
ARTIFACT_REGEX = re.compile(
    r"test-reports-test-(?P<name>\w+)-\d+-\d+-(?P<runner>[\w\.]+)_(?P<job>\d+).zip"
)


def upload_dynamo_perf_stats_to_rockset(
    repo: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    head_branch: str,
) -> List[Dict[str, Any]]:
    perf_stats = []
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        for artifact in ARTIFACTS:
            artifact_paths = download_s3_artifacts(
                artifact, workflow_run_id, workflow_run_attempt
            )

            # Unzip to get perf stats csv files
            for path in artifact_paths:
                m = ARTIFACT_REGEX.match(str(path))
                if not m:
                    print(f"Test report {path} has an invalid name. Skipping")
                    continue

                test_name = m.group("name")
                runner = m.group("runner")
                job_id = m.group("job")

                # Extract all files
                unzip(path)

                for csv_file in Path(".").glob("**/*.csv"):
                    filename = os.path.splitext(os.path.basename(csv_file))[0]
                    print(f"Processing {filename} from {path}")

                    with open(csv_file) as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=",")

                        for row in reader:
                            # If the row doesn't have a dev and a name column, it's not
                            # a torch dynamo perf stats csv file
                            if "dev" not in row or "name" not in row:
                                break

                            row.update(
                                {
                                    "workflow_id": workflow_run_id,  # type: ignore[dict-item]
                                    "run_attempt": workflow_run_attempt,  # type: ignore[dict-item]
                                    "test_name": test_name,
                                    "runner": runner,
                                    "job_id": job_id,
                                    "filename": filename,
                                    "head_branch": head_branch,
                                }
                            )
                            perf_stats.append(row)

                    # Done processing the file, removing it
                    os.remove(csv_file)

    return perf_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload dynamo perf stats from S3 to Rockset"
    )
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get perf stats from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )
    parser.add_argument(
        "--head-branch",
        type=str,
        required=True,
        help="Head branch of the workflow",
    )
    args = parser.parse_args()
    perf_stats = upload_dynamo_perf_stats_to_rockset(
        args.repo, args.workflow_run_id, args.workflow_run_attempt, args.head_branch
    )
    upload_to_rockset(
        collection="torch_dynamo_perf_stats",
        docs=perf_stats,
        workspace="inductor",
    )
