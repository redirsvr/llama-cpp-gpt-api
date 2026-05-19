package gpt

import (
    "context"

    "llama-cpp-gpt-api/internal/types"
    "llama-cpp-gpt-api/pkg/model"
)

const modelOwner = "local"

type ModelsLogic struct {
    ctx context.Context
}

func NewModelsLogic(ctx context.Context) *ModelsLogic {
    return &ModelsLogic{ctx: ctx}
}

func (l *ModelsLogic) ListModels() (*types.ResModelsList, error) {
    descs, err := model.ListDescriptors()
    if err != nil {
        return nil, err
    }
    data := make([]types.ModelObject, 0, len(descs))
    for _, d := range descs {
        data = append(data, toModelObject(d))
    }
    return &types.ResModelsList{
        Object: "list",
        Data:   data,
    }, nil
}

func (l *ModelsLogic) RetrieveModel(id string) (*types.ModelObject, error) {
    d, err := model.DescriptorByID(id)
    if err != nil {
        return nil, err
    }
    obj := toModelObject(d)
    return &obj, nil
}

func toModelObject(d model.ModelDescriptor) types.ModelObject {
    return types.ModelObject{
        ID:           d.ID,
        Object:       "model",
        Created:      d.Created,
        OwnedBy:      modelOwner,
        Capabilities: d.Capabilities,
    }
}
