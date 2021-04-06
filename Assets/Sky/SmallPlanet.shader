
Shader "Skybox/SmallPlanet"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _MainTex2 ("tex2D", 2D) = "white" {}
       _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
     }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            uniform sampler2D _MainTex2; 
            float4 _XYZPos, _SunDir;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

float2x2 rot(float a) { return float2x2(cos(a), sin(a), -sin(a), cos(a)); }

float2 map(float3 p) {
    float d = length(p) - 4.;
    return float2(d, 1.);
}

float3 calcNormal( in float3 p ) {
    const float h = 1e-4;
    float3 n = float3(0.0,0,0);
    for(int i = 0; i<4; i++) {
        float3 e = 0.5773*(2.0*float3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+e*h).x;
    }
    return normalize(n);
}

float2 rayProcess(float3 camPos, float3 rayDir, float start, float end) {
    float depth = start;
    for(int i = 0; i < 200; ++i) {
        float3 p = camPos + depth * rayDir;
        float2 dist = map(p);
        if(abs(dist.x) < 1e-3) return float2(depth, dist.y);
        depth += dist.x;
        if(dist.x >= end) break;
    }
    return float2(end, 0.);
}

float3 getBgnd(float3 d) {
    float3 w = abs(d);
    w = pow(w, float3(40.,40,40));
    w /= w.x + w.y + w.z;
    
    d *= 1.;
    float3 col = float3(0,0,0);
    col += w.x * tex2D(_MainTex2, d.yz + .5).rgb;
    col += w.y * tex2D(_MainTex2, d.zx + .5).rgb;
    col += w.z * tex2D(_MainTex2, d.xy + .5).rgb;
    
    return col;
}

float3 moontex2D(float2 uv) {
    float d = length(frac(uv) - .5);
    //return exp(-40. * d * d) * float3(1.);
    return tex2D(_MainTex, uv / 16.).rgb;
}

float heightmap(float2 uv) {
    return .2 * moontex2D(uv).r;
}

float3 normalmap(float2 p) {
    float2 e = float2(1e-3, 0);
    return normalize(float3(
        heightmap(p - e.xy) - heightmap(p + e.xy),
        heightmap(p - e.yx) - heightmap(p + e.yx),
        2. * e.x));
}

float3 triplanarNormal(float3 p, float3 nor, float3 w) {
    // compute rotation matrices for the 3 normal maps
    float3 xrY = cross(nor, float3(0,1,0));
    float3 xrX = cross(xrY, nor);
    float3x3 xrot = float3x3(xrX, sign(nor.x) * xrY, nor);

    float3 yrY = cross(nor, float3(0,0,1));
    float3 yrX = cross(yrY, nor);
    float3x3 yrot = float3x3(yrX, sign(nor.y) * xrY, nor);

    float3 zrY = cross(nor, float3(1,0,0));
    float3 zrX = cross(zrY, nor);
    float3x3 zrot = float3x3(zrX, sign(nor.z) * xrY, nor);

    float3 tnor = float3(0,0,0);
    tnor += w.x * mul(xrot , normalmap(p.yz + 5.));
    tnor += w.y * mul(yrot , normalmap(p.zx + float2(9., 14.)));
    tnor += w.z * mul(zrot , normalmap(p.xy + float2(12., 7.)));
    tnor = normalize(tnor);
    
    return tnor;
}

float3 shading(float3 ro, float3 rd) {
    float3 ld = normalize(_SunDir.xyz);  //normalize(float3(.5, 1, -.7));
    float3 sunCol = float3(1, .8, .5);
    float2 rp = rayProcess(ro, rd, 0., 40.);
    float3 p = ro + rp.x * rd;
    
    float3 col = float3(0,0,0);
    if(rp.y >= 1.) {
        float3 nor = calcNormal(p);
        
        float3 w = pow(abs(nor), float3(10.,10,10));
        w /= w.x + w.y + w.z;

        float3 tnor = triplanarNormal(p, nor, w);
        float3 mat = float3(0,0,0);
        mat += w.x * moontex2D(p.yz + 5.);
        mat += w.y * moontex2D(p.zx + float2(9., 14.));
        mat += w.z * moontex2D(p.xy + float2(12., 7.));
        
        // lighting
        float3 hal = normalize(ld - rd);
        float dif = max(dot(tnor, ld), 0.);
        
        // gspe is an atmospheric specular
        float spe = dot(nor, ld) * pow(max(dot(-rd, reflect(-ld, tnor)), 0.), 10.);
        float gspe = pow(max(dot(-rd, reflect(-ld, nor)) + .1, 0.), 7.);
        float cs = max(dot(ld, rd), 0.);
        
        col += mat * .01;
        col += mat *  dif * float3(1., .9, .8);
        col += .6 * lerp(spe, gspe, cs*cs) * sunCol;
        
        // simplfied fresnel
        float f = pow(1. - abs(dot(rd, nor)), 5.);
        col = lerp(col, pow(getBgnd(reflect(rd, nor)), float3(4.,4,4)), f);
        
    }
    else {
        // sky color
        col = pow(getBgnd(rd), float3(6.,6,6));
        
        float ldot = clamp(dot(rd, ld) + .001, 0., 1.);
        float sun = pow(ldot, 1000.);
        col = lerp(col * pow(1. - ldot*ldot, 3.), sunCol, sun);
        //col += spe * float3(.1, 0, 0);
    }
    
    return clamp(col, 0., 1.);
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin

//    float2 uv = (fragCoord - iResolution.xy / 2.) / iResolution.y;
//    float3 rd = normalize(float3(uv, -1.07));

//    float2 ang = iMouse.xy / iResolution.xy;
//    float yaw = 7. * ang.x + .2 * _Time.y;
//    float pitch = -5. * ang.y + 1. + .07 * _Time.y;

    float3 camPos = ro*.01; //+ float3(0., 0., 12);
//    camPos.yz *= rot(pitch); camPos.zx *= rot(yaw);
//    rd.yz     *= rot(pitch);     rd.zx *= rot(yaw);
    
    float3 col = shading(camPos, rd);
    fragColor = float4(pow(clamp(col, 0., 1.), float3(.4545,.4545,.4545)), 1);

                return fragColor;
            }

            ENDCG
        }
    }
}
