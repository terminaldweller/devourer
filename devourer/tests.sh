#!/usr/bin/env sh

curl -k -X GET "https://localhost:19019/mila/summ?url=https://dilipkumar.medium.com/standalone-mongodb-on-kubernetes-cluster-19e7b5896b27&summary=newspaper&audio=true"

curl -k -X GET "https://localhost:19019/mila/wiki?term=iommu&summary=none&audio=false"

curl -k -X GET "https://localhost:19019/mila/reqs?url=https://www.ietf.org/rfc/rfc2865.txt&sourcetype=text"

curl -k -X GET "https://localhost:19019/mila/pdf?feat=&url=https://www.rroij.com/open-access/mutation-testing-a-review-33-36.pdf"
curl -k -X GET "https://localhost:19019/mila/pdf?feat=refs&url=https://www.rroij.com/open-access/mutation-testing-a-review-33-36.pdf"
curl -k -X GET "https://localhost:19019/mila/pdf?feat=&url=https://www.rroij.com/open-access/mutation-testing-a-review-33-36.pdf&summarize=true"
curl -k -X GET "https://localhost:19019/mila/pdf?feat=keyword&url=https://www.rroij.com/open-access/mutation-testing-a-review-33-36.pdf&summarize=true"
